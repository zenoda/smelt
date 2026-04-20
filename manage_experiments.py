"""
实验管理工具

功能：
1. list  — 列出所有实验，显示关键指标
2. compare exp1 exp2 — 对比两个实验的配置和结果

实验记录格式：experiments/exp-{timestamp}.json
由 train.py 在训练结束后自动生成。

用法：
    python manage_experiments.py list
    python manage_experiments.py list --sort-by val_loss
    python manage_experiments.py compare exp-20260419_143025 exp-20260419_160012
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from config_utils import load_config, get


# ============================================================
# 实验记录加载
# ============================================================

def get_experiments_dir(cfg: dict = None) -> Path:
    """获取实验记录目录路径"""
    if cfg is None:
        cfg = load_config(cli_args=[])
    experiments_dir = get(cfg, "output.experiments_dir", "experiments")
    return Path(experiments_dir)


def load_experiment(path: Path) -> dict:
    """加载单个实验记录"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_experiments(experiments_dir: Path) -> list:
    """扫描实验目录，返回所有实验记录列表（按时间排序）"""
    if not experiments_dir.exists():
        return []

    experiments = []
    for f in sorted(experiments_dir.glob("exp-*.json")):
        try:
            exp = load_experiment(f)
            exp["_filename"] = f.name
            exp["_path"] = str(f)
            experiments.append(exp)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  警告: 无法读取 {f.name}: {e}")

    return experiments


# ============================================================
# 实验记录创建（供 train.py 调用）
# ============================================================

def create_experiment_record(
    config: dict,
    train_metrics: dict,
    eval_metrics: dict = None,
    training_duration_seconds: float = None,
    stopped_early: bool = False,
    early_stop_info: dict = None,
    adapter_dir: str = None,
    logging_dir: str = None,
    data_info: dict = None,
    experiments_dir: str = "experiments",
) -> str:
    """
    创建一个实验记录文件。

    参数：
        config: 完整配置快照（config/config.yaml 内容）
        train_metrics: 训练指标（trainer.train() 返回的 metrics）
        eval_metrics: 验证指标（trainer.evaluate() 返回的 metrics）
        training_duration_seconds: 训练耗时（秒）
        stopped_early: 是否早停
        early_stop_info: 早停详情 {"best_val_loss": ..., "best_step": ..., "patience": ...}
        adapter_dir: LoRA 适配器保存路径（用于链接到检查点）
        logging_dir: TensorBoard 日志目录路径
        data_info: 数据信息 {"total_samples": ..., "train_samples": ..., "val_samples": ..., "data_path": ...}
        experiments_dir: 实验记录保存目录

    返回：
        实验记录文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp-{timestamp}"

    # 获取 GPU 信息
    gpu_info = _get_gpu_info()

    # 构建实验记录
    record = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "training": {
            "duration_seconds": training_duration_seconds,
            "duration_human": _format_duration(training_duration_seconds),
            "stopped_early": stopped_early,
            "early_stop_info": early_stop_info,
            "metrics": {
                "train_loss": train_metrics.get("train_loss"),
                "train_steps": train_metrics.get("train_steps",
                               train_metrics.get("global_step")),
                "completed_epochs": train_metrics.get("epoch"),
                "train_runtime": train_metrics.get("train_runtime"),
                "train_samples_per_second": train_metrics.get(
                    "train_samples_per_second"),
            },
        },
        "evaluation": {},
        "data": data_info or {},
        "environment": {
            "gpu": gpu_info,
        },
        "artifacts": {
            "adapter_dir": adapter_dir,
            "tensorboard_dir": logging_dir,
        },
    }

    # 验证指标
    if eval_metrics:
        record["evaluation"] = {
            "val_loss": eval_metrics.get("eval_loss"),
            "eval_runtime": eval_metrics.get("eval_runtime"),
            "eval_samples_per_second": eval_metrics.get(
                "eval_samples_per_second"),
        }

    # 保存
    exp_dir = Path(experiments_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_path = exp_dir / f"{exp_id}.json"

    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    return str(exp_path)


def _get_gpu_info() -> dict:
    """获取 GPU 信息"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_mem
            gpu_mem_total_gb = gpu_mem_total / (1024**3)
            # 峰值使用量
            gpu_mem_peak = torch.cuda.max_memory_allocated(0)
            gpu_mem_peak_gb = gpu_mem_peak / (1024**3)
            return {
                "name": gpu_name,
                "vram_total_gb": round(gpu_mem_total_gb, 1),
                "vram_peak_gb": round(gpu_mem_peak_gb, 1),
            }
    except Exception:
        pass
    return {"name": "unknown", "vram_total_gb": 0, "vram_peak_gb": 0}


def _format_duration(seconds: float = None) -> str:
    """将秒数格式化为人类可读的时间"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ============================================================
# list 命令
# ============================================================

def cmd_list(args):
    """列出所有实验"""
    cfg = load_config(cli_args=[])
    experiments_dir = get_experiments_dir(cfg)
    experiments = list_experiments(experiments_dir)

    if not experiments:
        print(f"未找到实验记录。目录: {experiments_dir}")
        print("运行 python train.py 后会自动生成实验记录。")
        return

    # 排序
    sort_key = args.sort_by
    if sort_key == "time":
        experiments.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    elif sort_key == "val_loss":
        experiments.sort(
            key=lambda e: e.get("evaluation", {}).get("val_loss") or float("inf"),
        )
    elif sort_key == "train_loss":
        experiments.sort(
            key=lambda e: e.get("training", {}).get("metrics", {}).get("train_loss") or float("inf"),
        )
    elif sort_key == "duration":
        experiments.sort(
            key=lambda e: e.get("training", {}).get("duration_seconds") or float("inf"),
        )

    # 输出表头
    print(f"\n{'='*100}")
    print(f"实验列表 ({len(experiments)} 个)")
    print(f"{'='*100}")
    print(
        f"{'实验ID':<28} "
        f"{'训练损失':>10} "
        f"{'验证损失':>10} "
        f"{'Epochs':>8} "
        f"{'耗时':>12} "
        f"{'早停':>6} "
        f"{'LoRA r':>8} "
        f"{'LR':>12}"
    )
    print("-" * 100)

    for exp in experiments:
        exp_id = exp.get("experiment_id", "?")
        t_metrics = exp.get("training", {}).get("metrics", {})
        e_metrics = exp.get("evaluation", {})
        training = exp.get("training", {})
        config = exp.get("config", {})

        train_loss = t_metrics.get("train_loss")
        val_loss = e_metrics.get("val_loss")
        epochs = t_metrics.get("completed_epochs")
        duration = training.get("duration_human", "N/A")
        stopped_early = "是" if training.get("stopped_early") else "否"
        lora_r = get(config, "lora.r", "?")
        lr = get(config, "training.learning_rate", "?")

        print(
            f"{exp_id:<28} "
            f"{_fmt_float(train_loss):>10} "
            f"{_fmt_float(val_loss):>10} "
            f"{_fmt_val(epochs):>8} "
            f"{duration:>12} "
            f"{stopped_early:>6} "
            f"{str(lora_r):>8} "
            f"{_fmt_lr(lr):>12}"
        )

    print(f"{'='*100}")
    print(f"排序方式: {sort_key} | 目录: {experiments_dir}")


def _fmt_float(val, decimals=4):
    """格式化浮点数"""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_lr(val):
    """格式化学习率"""
    if val is None or val == "?":
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.1e}"
    return str(val)


def _fmt_val(val):
    """格式化通用值"""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


# ============================================================
# compare 命令
# ============================================================

def cmd_compare(args):
    """对比两个实验"""
    cfg = load_config(cli_args=[])
    experiments_dir = get_experiments_dir(cfg)

    exp1_id = args.exp1
    exp2_id = args.exp2

    # 查找实验文件（支持完整文件名或 exp-id）
    exp1_path = _find_experiment(experiments_dir, exp1_id)
    exp2_path = _find_experiment(experiments_dir, exp2_id)

    if exp1_path is None:
        print(f"错误: 找不到实验 '{exp1_id}'")
        print(f"可用实验: {', '.join(f.stem for f in experiments_dir.glob('exp-*.json'))}")
        sys.exit(1)
    if exp2_path is None:
        print(f"错误: 找不到实验 '{exp2_id}'")
        print(f"可用实验: {', '.join(f.stem for f in experiments_dir.glob('exp-*.json'))}")
        sys.exit(1)

    exp1 = load_experiment(exp1_path)
    exp2 = load_experiment(exp2_path)

    e1_id = exp1.get("experiment_id", exp1_id)
    e2_id = exp2.get("experiment_id", exp2_id)

    print(f"\n{'='*70}")
    print(f"实验对比")
    print(f"{'='*70}")
    print(f"  实验 A: {e1_id}")
    print(f"  实验 B: {e2_id}")
    print(f"{'='*70}")

    # 1. 训练指标对比
    print(f"\n--- 训练指标 ---")
    _compare_metrics(exp1, exp2, e1_id, e2_id)

    # 2. 关键配置差异
    print(f"\n--- 配置差异 ---")
    _compare_config(exp1, exp2, e1_id, e2_id)

    # 3. 环境信息
    print(f"\n--- 环境 ---")
    _compare_env(exp1, exp2, e1_id, e2_id)

    # 4. 数据信息
    print(f"\n--- 数据 ---")
    _compare_data(exp1, exp2, e1_id, e2_id)

    print(f"\n{'='*70}")


def _find_experiment(experiments_dir: Path, exp_id: str) -> Path:
    """
    根据 ID 查找实验文件。

    支持多种输入格式：
    - "exp-20260419_143025" → experiments/exp-20260419_143025.json
    - "exp-20260419_143025.json" → experiments/exp-20260419_143025.json
    - "20260419_143025" → experiments/exp-20260419_143025.json
    """
    if not experiments_dir.exists():
        return None

    # 直接文件名匹配
    candidates = [
        experiments_dir / f"{exp_id}.json",
        experiments_dir / exp_id,
        experiments_dir / f"exp-{exp_id}.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # 部分匹配
    for f in experiments_dir.glob("exp-*.json"):
        if exp_id in f.stem:
            return f

    return None


def _compare_metrics(exp1: dict, exp2: dict, e1_id: str, e2_id: str):
    """对比训练指标"""
    metrics_keys = [
        ("训练损失", "training.metrics.train_loss", True),
        ("验证损失", "evaluation.val_loss", True),
        ("完成 Epochs", "training.metrics.completed_epochs", False),
        ("训练步数", "training.metrics.train_steps", False),
        ("训练耗时", "training.duration_human", False),
        ("早停", "training.stopped_early", False),
    ]

    print(f"  {'指标':<16} {'A (' + e1_id[-15:] + ')':>20} {'B (' + e2_id[-15:] + ')':>20} {'差异':>15}")
    print(f"  {'-'*71}")

    for label, key_path, compute_diff in metrics_keys:
        val1 = _deep_get(exp1, key_path)
        val2 = _deep_get(exp2, key_path)

        v1_str = _fmt_metric(val1)
        v2_str = _fmt_metric(val2)

        diff_str = ""
        if compute_diff and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            pct = (diff / val1 * 100) if val1 != 0 else 0
            sign = "+" if diff > 0 else ""
            diff_str = f"{sign}{diff:.4f} ({sign}{pct:.1f}%)"

        print(f"  {label:<16} {v1_str:>20} {v2_str:>20} {diff_str:>15}")


def _compare_config(exp1: dict, exp2: dict, e1_id: str, e2_id: str):
    """对比关键配置参数"""
    config_keys = [
        ("LoRA rank", "config.lora.r"),
        ("LoRA alpha", "config.lora.alpha"),
        ("LoRA dropout", "config.lora.dropout"),
        ("学习率", "config.training.learning_rate"),
        ("Batch size", "config.training.per_device_train_batch_size"),
        ("梯度累积", "config.training.gradient_accumulation_steps"),
        ("训练 Epochs", "config.training.num_epochs"),
        ("Max steps", "config.training.max_steps"),
        ("Max seq len", "config.training.max_seq_length"),
        ("Warmup steps", "config.training.warmup_steps"),
        ("优化器", "config.training.optimizer"),
        ("验证集比例", "config.data.validation_split_ratio"),
        ("早停 patience", "config.early_stopping.patience"),
        ("早停 min_delta", "config.early_stopping.min_delta"),
    ]

    has_diff = False
    for label, key_path in config_keys:
        val1 = _deep_get(exp1, key_path)
        val2 = _deep_get(exp2, key_path)
        if val1 != val2:
            if not has_diff:
                print(f"  {'参数':<20} {'A':>20} {'B':>20}")
                print(f"  {'-'*60}")
                has_diff = True
            print(f"  {label:<20} {str(val1):>20} {str(val2):>20}")

    if not has_diff:
        print("  两个实验的关键配置完全一致。")


def _compare_env(exp1: dict, exp2: dict, e1_id: str, e2_id: str):
    """对比环境信息"""
    gpu1 = _deep_get(exp1, "environment.gpu", {})
    gpu2 = _deep_get(exp2, "environment.gpu", {})

    peak1 = gpu1.get("vram_peak_gb", "N/A")
    peak2 = gpu2.get("vram_peak_gb", "N/A")

    print(f"  GPU A: {gpu1.get('name', 'N/A')} (峰值 VRAM: {peak1} GB)")
    print(f"  GPU B: {gpu2.get('name', 'N/A')} (峰值 VRAM: {peak2} GB)")


def _compare_data(exp1: dict, exp2: dict, e1_id: str, e2_id: str):
    """对比数据信息"""
    data1 = exp1.get("data", {})
    data2 = exp2.get("data", {})

    data_keys = [
        ("总样本数", "total_samples"),
        ("训练集", "train_samples"),
        ("验证集", "val_samples"),
        ("数据文件", "data_path"),
    ]

    for label, key in data_keys:
        val1 = data1.get(key, "N/A")
        val2 = data2.get(key, "N/A")
        marker = " *" if val1 != val2 else ""
        print(f"  {label:<12} A: {val1}  |  B: {val2}{marker}")


def _deep_get(d: dict, dotted_key: str, default=None):
    """通过点号路径获取嵌套字典值"""
    keys = dotted_key.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fmt_metric(val):
    """格式化指标值"""
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "是" if val else "否"
    if isinstance(val, float):
        if abs(val) < 0.01:
            return f"{val:.6f}"
        return f"{val:.4f}"
    return str(val)


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验管理工具 — 列出和对比训练实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python manage_experiments.py list
  python manage_experiments.py list --sort-by val_loss
  python manage_experiments.py compare exp-20260419_143025 exp-20260419_160012
  python manage_experiments.py compare 20260419_143025 20260419_160012
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # list 子命令
    list_parser = subparsers.add_parser("list", help="列出所有实验")
    list_parser.add_argument(
        "--sort-by",
        choices=["time", "val_loss", "train_loss", "duration"],
        default="time",
        help="排序方式（默认按时间）",
    )

    # compare 子命令
    compare_parser = subparsers.add_parser("compare", help="对比两个实验")
    compare_parser.add_argument("exp1", help="实验 A 的 ID")
    compare_parser.add_argument("exp2", help="实验 B 的 ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
