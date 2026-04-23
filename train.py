"""
训练脚本：使用Unsloth QLoRA微调语言模型
支持任意兼容 Unsloth 的模型（如 Qwen、LLaMA、Mistral 等）

支持功能：
- 从 config/config.yaml 读取所有超参数（支持 CLI 覆盖）
- Epoch 为主要训练控制方式（num_epochs），max_steps 为可选硬性上限
- 训练/验证集分层划分（按类别比例保持一致）
- 每个 epoch 结束输出摘要：epoch 编号、train_loss、val_loss、已用时间
- TensorBoard 监控：train_loss, val_loss, learning_rate 实时曲线
- 验证损失停滞警告：连续 N 个评估周期 val_loss 未下降时输出警告
- 早停机制：验证损失连续 N 个评估周期未改善时自动停止训练，加载最优检查点
- 支持从检查点恢复训练，正确计算剩余 epoch
"""

import os
os.environ["UNSLOTH_TELEMETRY"] = "0"

import sys
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from config_utils import load_config, get, print_config
from manage_experiments import create_experiment_record


# ============================================================
# 类别推断（用于分层划分）
# ============================================================

def _infer_category(sample: dict) -> str:
    """
    从样本中推断类别。

    优先级：
    1. metadata.category（精确，来自 generate_data.py 合并输出）
    2. 内容推断（回退，兼容无 metadata 的旧数据）

    内容推断策略：
    - 包含 tool_call/tool_response XML → tool_calling
    - 包含前端关键词（Vue、ElementPlus、Composition API 等）→ frontend_dev
    - 默认 → java_coding
    """
    # 优先使用 metadata.category
    category = sample.get("metadata", {}).get("category", "")
    if category:
        return category

    # 回退到内容推断
    messages = sample.get("messages", [])
    full_text = ""
    system_text = ""
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        full_text += content
        if msg.get("role") == "system":
            system_text += content

    # 工具调用类别：检查是否包含 tool_call XML 标记
    if "<tool_call>" in full_text or "<tool_response>" in full_text:
        return "tool_calling"

    # 前端开发类别：检测前端相关关键词
    frontend_keywords = ["Vue", "ElementPlus", "Element Plus", "前端",
                         "组件", "Composition API", "Pinia", "Vite",
                         "defineProps", "defineEmits", "script setup",
                         "el-table", "el-form", "axios"]
    combined_text = system_text + full_text
    if any(kw.lower() in combined_text.lower() for kw in frontend_keywords):
        return "frontend_dev"

    # 默认：Java 编程
    return "java_coding"


def stratified_split(
    samples: list,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple:
    """
    对样本列表进行分层划分，保持各类别在训练集和验证集中的比例一致。

    参数：
        samples: 样本列表，每个元素是 {"messages": [...]} 字典
        val_ratio: 验证集占比（0.0-1.0）
        seed: 随机种子（确保可复现）

    返回：
        (train_samples, val_samples) 两个列表
    """
    if val_ratio <= 0 or val_ratio >= 1:
        print(f"  验证集比例 {val_ratio} 无效，跳过划分")
        return samples, []

    # 按类别分组
    by_category = defaultdict(list)
    for sample in samples:
        cat = _infer_category(sample)
        by_category[cat].append(sample)

    print(f"\n  数据类别分布:")
    for cat, cat_samples in sorted(by_category.items()):
        print(f"    {cat}: {len(cat_samples)} 条")

    # 分层划分
    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for cat, cat_samples in by_category.items():
        # 打乱类别内的顺序
        shuffled = list(cat_samples)
        rng.shuffle(shuffled)

        # 计算验证集数量（至少1条，除非该类别样本数 < 2）
        n_val = max(1, int(len(shuffled) * val_ratio))
        if len(shuffled) < 2:
            # 样本太少，全部放入训练集
            train_samples.extend(shuffled)
            continue

        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    # 最终打乱（保持可复现）
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


# ============================================================
# TensorBoard 监控回调
# ============================================================

class ValLossStagnationCallback(TrainerCallback):
    """
    自定义回调：监控验证损失，当连续 N 个评估周期 val_loss 未下降时输出警告。

    参数：
        patience: 连续多少个评估周期 val_loss 不下降时发出警告（默认 50 步）
    """

    def __init__(self, patience: int = 50):
        self.patience = patience
        self.best_val_loss = float("inf")
        self.stagnation_count = 0
        self.eval_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """每次评估后检查 val_loss 是否改善"""
        if metrics is None:
            return

        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        self.eval_count += 1

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= self.patience:
                print(
                    f"\n⚠️  警告: 验证损失已连续 {self.stagnation_count} 个评估周期未下降！"
                    f" (当前 val_loss={val_loss:.4f}, 最佳 val_loss={self.best_val_loss:.4f})"
                    f"\n   可能存在过拟合，请关注 TensorBoard 曲线。"
                )


class EarlyStoppingCallback(TrainerCallback):
    """
    早停回调：当验证损失连续 patience 个评估周期未改善时停止训练。

    工作原理：
    1. 每次评估后检查 eval_loss 是否比历史最优改善了 min_delta 以上
    2. 如果改善了，重置计数器并记录当前为最佳检查点
    3. 如果未改善，计数器+1；达到 patience 时设置 should_training_stop=True
    4. 训练结束后，main() 函数根据 stopped_early 标志加载最佳检查点

    参数：
        patience: 连续多少个评估周期 val_loss 不改善时停止（默认 3）
        min_delta: 最小改善幅度，损失下降小于此值不算改善（默认 0.001）
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.best_step = 0
        self.stopped_early = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """每次评估后检查是否应该早停"""
        if metrics is None:
            return

        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if val_loss < self.best_val_loss - self.min_delta:
            # 有足够改善
            self.best_val_loss = val_loss
            self.no_improve_count = 0
            self.best_step = state.global_step
            print(
                f"  [早停] val_loss 改善至 {val_loss:.4f} (step {state.global_step})"
            )
        else:
            self.no_improve_count += 1
            remaining = self.patience - self.no_improve_count
            print(
                f"  [早停] val_loss 未改善 ({val_loss:.4f} vs 最佳 {self.best_val_loss:.4f})，"
                f"剩余耐心: {remaining}/{self.patience}"
            )
            if self.no_improve_count >= self.patience:
                print(
                    f"\n{'='*60}"
                    f"\n  早停触发！验证损失连续 {self.patience} 个评估周期未改善。"
                    f"\n  最佳 val_loss: {self.best_val_loss:.4f} (step {self.best_step})"
                    f"\n{'='*60}"
                )
                self.stopped_early = True
                control.should_training_stop = True


class EpochSummaryCallback(TrainerCallback):
    """
    自定义回调：在每个 epoch 结束时输出 epoch 编号、train_loss、val_loss 和已用时间。

    工作原理：
    - on_train_begin: 记录训练开始时间
    - on_epoch_end: 输出 epoch 摘要（train_loss 来自 Trainer 日志）
    - on_evaluate: 记录最新的 val_loss（供 on_epoch_end 使用）
    """

    def __init__(self):
        self.train_start_time = None
        self.latest_val_loss = None
        self.epoch_train_losses = []

    def on_train_begin(self, args, state, control, **kwargs):
        """记录训练开始时间"""
        self.train_start_time = datetime.now()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """收集训练过程中的 loss 记录"""
        if logs and "loss" in logs:
            self.epoch_train_losses.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """记录最新验证损失"""
        if metrics and "eval_loss" in metrics:
            self.latest_val_loss = metrics["eval_loss"]

    def on_epoch_end(self, args, state, control, **kwargs):
        """在每个 epoch 结束时输出摘要"""
        epoch = int(state.epoch) if state.epoch else 0
        elapsed = datetime.now() - self.train_start_time if self.train_start_time else None
        elapsed_str = str(elapsed).split(".")[0] if elapsed else "N/A"

        # 计算本 epoch 的平均 train_loss
        if self.epoch_train_losses:
            avg_train_loss = sum(self.epoch_train_losses) / len(self.epoch_train_losses)
            train_loss_str = f"{avg_train_loss:.4f}"
        else:
            train_loss_str = "N/A"

        val_loss_str = f"{self.latest_val_loss:.4f}" if self.latest_val_loss is not None else "N/A"

        print(
            f"\n{'─'*50}"
            f"\n  Epoch {epoch} 完成"
            f"\n  train_loss: {train_loss_str}"
            f"\n  val_loss:   {val_loss_str}"
            f"\n  已用时间:   {elapsed_str}"
            f"\n{'─'*50}"
        )

        # 重置 epoch 级别的 loss 收集
        self.epoch_train_losses = []


def build_logging_dir(base_dir: str = "runs", experiment_name: str = "") -> str:
    """
    构建 TensorBoard 日志目录路径。

    格式：{base_dir}/{experiment_name}_{YYYYMMDD_HHMMSS}
    例如：runs/train_20260419_143025

    参数：
        base_dir: 日志根目录（默认 runs/）
        experiment_name: 实验名称前缀（默认空，使用 "train"）

    返回：
        完整的日志目录路径
    """
    if not experiment_name:
        experiment_name = "train"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(base_dir) / f"{experiment_name}_{timestamp}")


# ============================================================
# 模型与数据加载
# ============================================================

def load_model_and_tokenizer(
    model_name: str = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    load_in_16bit: bool = False,
    full_finetuning: bool = False,
):
    """加载模型和分词器"""
    if model_name is None:
        raise ValueError("必须指定模型路径（通过 config/config.yaml 的 model.path 或 CLI --model.path 参数）")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        full_finetuning=full_finetuning,
        dtype=None,
        local_files_only=True,
    )
    return model, tokenizer


def apply_lora(
    model,
    r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0,
    target_modules: list = None,
    max_seq_length: int = 4096,
):
    """应用LoRA适配器"""
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    return model


def load_and_split_data(
    data_path: str = "training_data.jsonl",
    val_ratio: float = 0.10,
    split_seed: int = 42,
) -> tuple:
    """
    加载训练数据 JSONL 文件，并进行分层划分。

    返回：
        (train_samples, val_samples) — 原始 JSON 对象列表
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if "messages" not in sample:
                    print(f"  警告: 第{line_num}行缺少messages字段，跳过")
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"  警告: 第{line_num}行JSON解析失败: {e}，跳过")

    print(f"  共加载 {len(samples)} 条样本")

    if val_ratio > 0 and len(samples) >= 2:
        train_samples, val_samples = stratified_split(samples, val_ratio, split_seed)
        print(f"\n  划分结果: 训练集 {len(train_samples)} 条, 验证集 {len(val_samples)} 条")
    else:
        train_samples = samples
        val_samples = []
        if val_ratio > 0:
            print("  样本数不足，跳过验证集划分")

    return train_samples, val_samples


def format_dataset(tokenizer, samples: list) -> Dataset:
    """
    将样本列表格式化为 HuggingFace Dataset，使用 tokenizer.apply_chat_template。
    """
    if not samples:
        return None

    dataset = Dataset.from_dict({"messages": [s["messages"] for s in samples]})

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            # 使用tokenizer自带的chat template
            # enable_thinking=False：训练数据不含thinking，避免插入空think块
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    max_seq_length: int = 4096,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 20,
    num_epochs: int = 3,
    max_steps: int = -1,
    learning_rate: float = 5e-5,
    output_dir: str = "outputs",
    logging_steps: int = 1,
    seed: int = 3407,
    save_steps: int = 100,
    save_total_limit: int = 3,
    optimizer: str = "adamw_8bit",
    dataset_num_proc: int = 2,
    logging_dir: str = None,
    callbacks: list = None,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    save_strategy: str = None,
):
    """
    创建训练器，支持可选的验证集、TensorBoard 监控、早停和 epoch/step 双重控制。

    训练控制逻辑：
    - num_epochs: 训练轮数（默认 3），以此作为主要控制方式
    - max_steps: 可选硬性上限（默认 -1 表示不限制）
    - 当两者同时设置时，以先达到的为准（HuggingFace Trainer 内置行为）
    - 恢复训练：通过 trainer.train(resume_from_checkpoint=...) 实现
    """

    # 构建 SFTConfig 参数
    sft_kwargs = dict(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        output_dir=output_dir,
        optim=optimizer,
        seed=seed,
        dataset_num_proc=dataset_num_proc,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
    )

    # max_steps > 0 时作为硬性上限；-1 表示不限制（由 num_epochs 控制）
    if max_steps > 0:
        sft_kwargs["max_steps"] = max_steps

    # TensorBoard 配置
    if logging_dir is not None:
        sft_kwargs["report_to"] = "tensorboard"
        # transformers>=5.2 弃用了 logging_dir 参数，改用环境变量
        os.environ["TENSORBOARD_LOGGING_DIR"] = logging_dir

    # 如果有验证集，启用 epoch 级别评估
    if eval_dataset is not None:
        sft_kwargs["eval_strategy"] = "epoch"
        sft_kwargs["per_device_eval_batch_size"] = per_device_train_batch_size

    # 早停支持：启用 load_best_model_at_end 时，需要同步 save 和 eval 策略
    if load_best_model_at_end:
        sft_kwargs["load_best_model_at_end"] = True
        sft_kwargs["metric_for_best_model"] = metric_for_best_model
        sft_kwargs["greater_is_better"] = greater_is_better
        # save_strategy 必须与 eval_strategy 一致才能在评估时保存检查点
        if eval_dataset is not None:
            sft_kwargs["save_strategy"] = "epoch"

    # 允许外部覆盖 save_strategy
    if save_strategy is not None:
        sft_kwargs["save_strategy"] = save_strategy

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(**sft_kwargs),
        callbacks=callbacks,
    )
    return trainer


# ============================================================
# 主流程
# ============================================================

def main():
    # 加载配置（支持 CLI 覆盖）
    cfg = load_config(cli_args=sys.argv[1:])

    # 读取配置项
    MODEL_NAME = get(cfg, "model.path", None)
    if MODEL_NAME is None:
        print("\n错误: 必须在 config/config.yaml 的 model.path 中指定基础模型路径！")
        print("请先复制配置模板: cp config/config.example.yaml config/config.yaml")
        print("然后编辑 model.path 为你的模型路径")
        sys.exit(1)
    LOAD_IN_4BIT = get(cfg, "model.load_in_4bit", True)
    LOAD_IN_16BIT = get(cfg, "model.load_in_16bit", False)
    FULL_FINETUNING = get(cfg, "model.full_finetuning", False)

    LORA_R = get(cfg, "lora.r", 32)
    LORA_ALPHA = get(cfg, "lora.alpha", 32)
    LORA_DROPOUT = get(cfg, "lora.dropout", 0.0)
    TARGET_MODULES = get(cfg, "lora.target_modules", None)

    MAX_SEQ_LENGTH = get(cfg, "training.max_seq_length", 4096)
    BATCH_SIZE = get(cfg, "training.per_device_train_batch_size", 2)
    GRADIENT_ACCUMULATION = get(cfg, "training.gradient_accumulation_steps", 4)
    NUM_EPOCHS = get(cfg, "training.num_epochs", 3)
    MAX_STEPS = get(cfg, "training.max_steps", -1)
    LEARNING_RATE = get(cfg, "training.learning_rate", 5e-5)
    WARMUP_STEPS = get(cfg, "training.warmup_steps", 20)
    LOGGING_STEPS = get(cfg, "training.logging_steps", 1)
    SAVE_STEPS = get(cfg, "training.save_steps", 100)
    SAVE_TOTAL_LIMIT = get(cfg, "training.save_total_limit", 3)
    OPTIMIZER = get(cfg, "training.optimizer", "adamw_8bit")
    SEED = get(cfg, "training.seed", 3407)
    DATASET_NUM_PROC = get(cfg, "training.dataset_num_proc", 2)

    DATA_PATH = get(cfg, "data.training_data_path", "training_data.jsonl")
    VAL_RATIO = get(cfg, "data.validation_split_ratio", 0.10)
    SPLIT_SEED = get(cfg, "data.split_seed", 42)

    OUTPUT_DIR = get(cfg, "output.checkpoint_dir", "outputs")
    ADAPTER_DIR = get(cfg, "output.adapter_dir", "lora_adapter")
    MERGED_DIR = get(cfg, "output.merged_dir", "merged_model")
    MERGE_MODEL = get(cfg, "output.merge_model", False)
    MERGE_METHOD = get(cfg, "output.merge_method", "merged_16bit")

    # TensorBoard 配置
    TB_DIR = get(cfg, "output.tensorboard_dir", "runs")
    EXPERIMENT_NAME = get(cfg, "training.experiment_name", "train")
    VAL_STAGNATION_PATIENCE = get(cfg, "training.val_loss_stagnation_patience", 50)

    # 早停配置
    EARLY_STOPPING_ENABLED = get(cfg, "early_stopping.enabled", True)
    EARLY_STOPPING_PATIENCE = get(cfg, "early_stopping.patience", 3)
    EARLY_STOPPING_MIN_DELTA = get(cfg, "early_stopping.min_delta", 0.001)

    # 构建 TensorBoard 日志目录
    logging_dir = build_logging_dir(base_dir=TB_DIR, experiment_name=EXPERIMENT_NAME)

    print("=" * 60)
    print("QLoRA 微调训练")
    print(f"  模型: {MODEL_NAME}")
    print(f"  QLoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"  序列长度: {MAX_SEQ_LENGTH}")
    print(f"  有效batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    if MAX_STEPS > 0:
        print(f"  最大步数上限: {MAX_STEPS}（与 epoch 取先到者）")
    else:
        print(f"  最大步数: 不限制（由 epoch 控制）")
    print(f"  验证集比例: {VAL_RATIO}")
    print(f"  划分种子: {SPLIT_SEED}")
    print(f"  合并模型: {'是' if MERGE_MODEL else '否'}")
    print(f"  TensorBoard: {logging_dir}")
    if EARLY_STOPPING_ENABLED:
        print(f"  早停: 启用 (patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA})")
    else:
        print(f"  早停: 禁用")
    print("=" * 60)

    # 检查训练数据文件
    if not Path(DATA_PATH).exists():
        print(f"\n错误: 训练数据文件 {DATA_PATH} 不存在！")
        print("请先运行 python generate_data.py --merge 生成训练数据")
        sys.exit(1)

    print("\n加载模型...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_16bit=LOAD_IN_16BIT,
        full_finetuning=FULL_FINETUNING,
    )

    print("\n应用LoRA适配器...")
    model = apply_lora(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("\n加载训练数据...")
    train_samples, val_samples = load_and_split_data(
        data_path=DATA_PATH,
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
    )

    print("\n格式化训练集...")
    train_dataset = format_dataset(tokenizer, train_samples)
    print(f"训练集样本数量: {len(train_dataset)}")

    eval_dataset = None
    if val_samples:
        print("格式化验证集...")
        eval_dataset = format_dataset(tokenizer, val_samples)
        print(f"验证集样本数量: {len(eval_dataset)}")

    # 打印一条样本确认格式
    print("\n训练集样本示例（前500字符）:")
    print(train_dataset[0]["text"][:500])
    print("...")

    # 构建回调列表
    callbacks = []
    early_stopping_cb = None

    # Epoch 摘要回调（始终启用）
    epoch_summary_cb = EpochSummaryCallback()
    callbacks.append(epoch_summary_cb)

    if eval_dataset is not None:
        stagnation_cb = ValLossStagnationCallback(patience=VAL_STAGNATION_PATIENCE)
        callbacks.append(stagnation_cb)
        print(f"\n  验证损失停滞监控: 连续 {VAL_STAGNATION_PATIENCE} 个评估周期不下降时警告")

        if EARLY_STOPPING_ENABLED:
            early_stopping_cb = EarlyStoppingCallback(
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
            )
            callbacks.append(early_stopping_cb)
            print(f"  早停回调: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}")
    elif EARLY_STOPPING_ENABLED:
        print("\n  注意: 早停已启用但无验证集，早停不会生效")

    # 是否启用 load_best_model_at_end（早停需要此功能加载最优检查点）
    use_load_best = EARLY_STOPPING_ENABLED and eval_dataset is not None

    # 检测是否有可恢复的检查点
    resume_checkpoint = None
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        checkpoints = sorted(
            [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"\n  发现检查点: {resume_checkpoint}，将从此处恢复训练")

    print("\n创建训练器...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optimizer=OPTIMIZER,
        dataset_num_proc=DATASET_NUM_PROC,
        logging_dir=logging_dir,
        callbacks=callbacks if callbacks else None,
        load_best_model_at_end=use_load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if eval_dataset is not None:
        print("  验证策略: 每个 epoch 结束时评估")
        if use_load_best:
            print("  最优模型: 训练结束后自动加载验证损失最低的检查点")

    print("\n" + "=" * 60)
    print("开始训练...")
    if resume_checkpoint:
        print(f"  从检查点恢复: {resume_checkpoint}")
    print("=" * 60)

    train_start_time = datetime.now()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_end_time = datetime.now()
    training_duration_seconds = (train_end_time - train_start_time).total_seconds()

    # 输出训练结果摘要
    print("\n" + "=" * 60)
    metrics = train_result.metrics
    train_steps = metrics.get('train_steps', metrics.get('global_step', 'N/A'))
    completed_epochs = metrics.get('epoch', 'N/A')

    # 判断训练结束方式
    stopped_early = early_stopping_cb is not None and early_stopping_cb.stopped_early
    if stopped_early:
        print("训练完成（早停触发）")
        print(f"  早停原因: 验证损失连续 {EARLY_STOPPING_PATIENCE} 个评估周期未改善")
        print(f"  最佳 val_loss: {early_stopping_cb.best_val_loss:.4f} (step {early_stopping_cb.best_step})")
        print(f"  已自动加载最优检查点")
    elif MAX_STEPS > 0 and isinstance(train_steps, (int, float)) and train_steps >= MAX_STEPS:
        print(f"训练完成（达到最大步数上限 {MAX_STEPS}）")
    else:
        print(f"训练完成（正常结束，{NUM_EPOCHS} 个 epoch）")

    print(f"  完成 epoch: {completed_epochs}")
    print(f"  训练损失: {metrics.get('train_loss', 'N/A')}")
    print(f"  训练步数: {train_steps}")

    # 如果有验证集，运行最终评估
    eval_metrics = None
    if eval_dataset is not None:
        print("\n运行最终验证评估...")
        import gc, torch as _torch
        gc.collect()
        _torch.cuda.empty_cache()
        _torch.cuda.reset_peak_memory_stats()
        try:
            eval_metrics = trainer.evaluate()
            final_val_loss = eval_metrics.get('eval_loss', 'N/A')
            print(f"  验证损失: {final_val_loss}")
            if stopped_early:
                print(f"  （此为最优检查点的验证损失）")
        except Exception as e:
            print(f"  [警告] 最终验证评估失败: {e}")
            print(f"  训练已完成，模型不受影响。跳过最终评估。")

    print(f"\n  TensorBoard 日志: {logging_dir}")
    print(f"  查看训练曲线: tensorboard --logdir {TB_DIR}")
    print("=" * 60)

    print("\n保存模型...")

    # 保存LoRA适配器
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"LoRA适配器已保存到: {ADAPTER_DIR}/")

    # 可选：保存合并后的模型
    if MERGE_MODEL:
        import psutil

        print(f"\n合并模型到 {MERGED_DIR}（方式: {MERGE_METHOD}）...")
        ram_before = psutil.virtual_memory()
        print(f"  合并前 RAM: 已用 {ram_before.used / (1024**3):.1f}GB / 总计 {ram_before.total / (1024**3):.1f}GB")

        model.save_pretrained_merged(
            MERGED_DIR,
            tokenizer,
            save_method=MERGE_METHOD,
        )

        ram_after = psutil.virtual_memory()
        ram_peak_used = ram_after.used / (1024**3)
        ram_total = ram_after.total / (1024**3)
        print(f"  合并后 RAM: 已用 {ram_peak_used:.1f}GB / 总计 {ram_total:.1f}GB")
        if ram_peak_used > 58:  # 接近 64GB 时警告
            print(f"  警告: RAM 使用较高 ({ram_peak_used:.1f}GB)，接近 64GB 上限")
        print(f"合并模型已保存到: {MERGED_DIR}/")
    else:
        print("跳过模型合并（merge_model=false）")

    # 生成实验记录
    EXPERIMENTS_DIR = get(cfg, "output.experiments_dir", "experiments")

    # 数据信息
    data_info = {
        "data_path": DATA_PATH,
        "total_samples": len(train_samples) + len(val_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }

    # 早停信息
    early_stop_info_dict = None
    if stopped_early and early_stopping_cb is not None:
        early_stop_info_dict = {
            "best_val_loss": early_stopping_cb.best_val_loss,
            "best_step": early_stopping_cb.best_step,
            "patience": EARLY_STOPPING_PATIENCE,
        }

    exp_path = create_experiment_record(
        config=cfg,
        train_metrics=metrics,
        eval_metrics=eval_metrics,
        training_duration_seconds=training_duration_seconds,
        stopped_early=stopped_early,
        early_stop_info=early_stop_info_dict,
        adapter_dir=ADAPTER_DIR,
        logging_dir=logging_dir,
        data_info=data_info,
        experiments_dir=EXPERIMENTS_DIR,
    )
    print(f"\n实验记录已保存: {exp_path}")
    print(f"  查看所有实验: python manage_experiments.py list")
    print(f"  对比实验: python manage_experiments.py compare <exp1> <exp2>")

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
