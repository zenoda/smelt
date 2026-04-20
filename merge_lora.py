"""
多 LoRA 适配器合并脚本

功能：
- 将多个独立训练的 LoRA 适配器合并为一个新适配器
- 支持合并方法：linear（加权平均）、ties（TIES-Merging）、dare_ties（DARE+TIES）、cat（拼接合并）
- 合并参数从 config/config.yaml 的 lora_merge 段读取
- 自动验证所有适配器是否基于相同的基础模型训练
- 合并输出为新的 LoRA 适配器，可直接用于推理或进一步合并到基础模型

用法：
    python merge_lora.py                          # 使用 config/config.yaml 配置
    python merge_lora.py --lora_merge.method ties  # CLI 覆盖合并方法
    python merge_lora.py --lora_merge.density 0.3  # CLI 覆盖稀疏化密度
"""

import os
os.environ["UNSLOTH_TELEMETRY"] = "0"

import sys
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from config_utils import load_config, get, print_config


# ============================================================
# 支持的合并方法
# ============================================================
SUPPORTED_METHODS = {
    "linear":    "加权线性平均（最简单稳定）",
    "ties":      "TIES-Merging（稀疏化后合并，保留显著参数）",
    "dare_ties": "DARE + TIES（随机丢弃 + TIES，更激进的稀疏化）",
    "cat":       "拼接合并（增大适配器维度，rank = sum of all ranks）",
}


# ============================================================
# 适配器验证
# ============================================================

def load_adapter_config(adapter_path: str) -> Dict:
    """加载适配器的 adapter_config.json。"""
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"适配器目录中未找到 adapter_config.json: {adapter_path}\n"
            f"请确认路径是否为有效的 LoRA 适配器目录。"
        )
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_adapters(adapter_paths: List[str]) -> Tuple[str, List[Dict]]:
    """
    验证所有适配器是否基于相同的基础模型训练。

    返回：
        (base_model_path, adapter_configs_list)

    异常：
        ValueError: 适配器不兼容时抛出
    """
    if len(adapter_paths) < 2:
        raise ValueError(
            f"合并至少需要 2 个适配器，当前只有 {len(adapter_paths)} 个。\n"
            f"请在 config/config.yaml 的 lora_merge.adapters 中配置至少 2 个适配器路径。"
        )

    configs = []
    base_models = set()
    peft_types = set()

    for path in adapter_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"适配器路径不存在: {path}")

        config = load_adapter_config(path)
        configs.append(config)

        base_model = config.get("base_model_name_or_path", "")
        peft_type = config.get("peft_type", "")

        base_models.add(base_model)
        peft_types.add(peft_type)

    # 验证：所有适配器必须基于相同的基础模型
    if len(base_models) > 1:
        details = "\n".join(
            f"  - {path}: {configs[i].get('base_model_name_or_path', 'unknown')}"
            for i, path in enumerate(adapter_paths)
        )
        raise ValueError(
            f"适配器基于不同的基础模型，无法合并：\n{details}\n\n"
            f"所有适配器必须从相同的基础模型训练才能合并。"
        )

    # 验证：所有适配器必须是 LoRA 类型
    if peft_types != {"LORA"}:
        raise ValueError(
            f"仅支持 LoRA 类型适配器合并。检测到的类型: {peft_types}"
        )

    base_model_path = base_models.pop()
    print(f"[验证通过] 所有 {len(adapter_paths)} 个适配器均基于相同的基础模型:")
    print(f"  基础模型: {base_model_path}")
    print()

    # 输出适配器摘要
    for i, (path, config) in enumerate(zip(adapter_paths, configs)):
        rank = config.get("r", "?")
        alpha = config.get("lora_alpha", "?")
        targets = config.get("target_modules", [])
        print(f"  适配器 {i+1}: {path}")
        print(f"    rank={rank}, alpha={alpha}, targets={len(targets)} modules")

    print()
    return base_model_path, configs


def validate_weights(weights: List[float], adapter_count: int) -> List[float]:
    """验证并规范化权重列表。"""
    if len(weights) != adapter_count:
        raise ValueError(
            f"权重数量 ({len(weights)}) 与适配器数量 ({adapter_count}) 不匹配。\n"
            f"请确保 lora_merge.weights 列表与 lora_merge.adapters 列表长度相同。"
        )

    if all(w == 0 for w in weights):
        raise ValueError("所有权重均为 0，无法合并。")

    return weights


# ============================================================
# 合并核心逻辑
# ============================================================

def merge_adapters(
    base_model_path: str,
    adapter_paths: List[str],
    weights: List[float],
    method: str,
    density: float,
    output_path: str,
    cfg: Dict,
) -> None:
    """
    执行多 LoRA 适配器合并。

    步骤：
    1. 加载基础模型（4-bit 量化）
    2. 加载第一个适配器作为 PeftModel
    3. 加载其余适配器
    4. 调用 add_weighted_adapter 进行合并
    5. 保存合并后的适配器
    """
    from peft import PeftModel
    from unsloth import FastLanguageModel

    load_in_4bit = get(cfg, "model.load_in_4bit", True)
    max_seq_length = get(cfg, "training.max_seq_length", 4096)

    # ----------------------------------------------------------
    # Step 1: 加载基础模型
    # ----------------------------------------------------------
    print("=" * 60)
    print("  Step 1: 加载基础模型")
    print("=" * 60)
    print(f"  路径: {base_model_path}")
    print(f"  4-bit 量化: {load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  基础模型加载完成，GPU VRAM: {vram_used:.2f} GB")
    print()

    # ----------------------------------------------------------
    # Step 2: 加载第一个适配器
    # ----------------------------------------------------------
    print("=" * 60)
    print("  Step 2: 加载适配器")
    print("=" * 60)

    adapter_names = []
    first_adapter = adapter_paths[0]
    first_name = f"adapter_0"
    adapter_names.append(first_name)

    print(f"  加载适配器 1/{len(adapter_paths)}: {first_adapter} (名称: {first_name})")
    model = PeftModel.from_pretrained(
        model,
        first_adapter,
        adapter_name=first_name,
        is_trainable=False,
    )

    # ----------------------------------------------------------
    # Step 3: 加载其余适配器
    # ----------------------------------------------------------
    for i, adapter_path in enumerate(adapter_paths[1:], start=1):
        name = f"adapter_{i}"
        adapter_names.append(name)
        print(f"  加载适配器 {i+1}/{len(adapter_paths)}: {adapter_path} (名称: {name})")
        model.load_adapter(adapter_path, adapter_name=name)

    print(f"\n  已加载 {len(adapter_names)} 个适配器: {adapter_names}")
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  当前 GPU VRAM: {vram_used:.2f} GB")
    print()

    # ----------------------------------------------------------
    # Step 4: 合并适配器
    # ----------------------------------------------------------
    print("=" * 60)
    print("  Step 3: 合并适配器")
    print("=" * 60)
    print(f"  合并方法: {method} ({SUPPORTED_METHODS[method]})")
    print(f"  权重: {weights}")
    if method in ("ties", "dare_ties"):
        print(f"  稀疏化密度: {density}")
    print()

    merged_name = "merged_adapter"

    # 构建 add_weighted_adapter 的参数
    merge_kwargs = {
        "adapters": adapter_names,
        "weights": weights,
        "adapter_name": merged_name,
        "combination_type": method,
    }

    # ties/dare_ties 需要 density 参数
    if method in ("ties", "dare_ties"):
        merge_kwargs["density"] = density
        merge_kwargs["majority_sign_method"] = "total"

    print("  正在合并...")
    model.base_model.add_weighted_adapter(**merge_kwargs)
    print("  合并完成！")

    # 设置合并后的适配器为活跃适配器
    model.set_adapter(merged_name)
    print(f"  已切换到合并适配器: {merged_name}")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  当前 GPU VRAM: {vram_used:.2f} GB")
    print()

    # ----------------------------------------------------------
    # Step 5: 保存合并后的适配器
    # ----------------------------------------------------------
    print("=" * 60)
    print("  Step 4: 保存合并后的适配器")
    print("=" * 60)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存适配器权重和配置
    model.save_pretrained(output_dir, selected_adapters=[merged_name])

    # 保存 tokenizer（方便后续直接使用）
    tokenizer.save_pretrained(output_dir)

    # 保存合并元信息
    merge_info = {
        "merge_method": method,
        "source_adapters": adapter_paths,
        "weights": weights,
        "density": density if method in ("ties", "dare_ties") else None,
        "base_model": base_model_path,
        "merged_adapter_name": merged_name,
        "peft_version": None,
    }

    try:
        import peft
        merge_info["peft_version"] = peft.__version__
    except Exception:
        pass

    merge_info_path = output_dir / "merge_info.json"
    with open(merge_info_path, "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)

    print(f"  适配器已保存到: {output_dir}")
    print(f"  合并信息已保存到: {merge_info_path}")

    # 列出输出文件
    total_size = 0
    for file in sorted(output_dir.iterdir()):
        if file.is_file():
            size_mb = file.stat().st_size / 1024**2
            total_size += size_mb
            print(f"    {file.name}: {size_mb:.1f} MB")

    print(f"  总大小: {total_size:.1f} MB")
    print()

    # ----------------------------------------------------------
    # 清理
    # ----------------------------------------------------------
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  GPU 内存已释放。")


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口：读取配置、验证、执行合并。"""
    # 加载配置
    cfg = load_config(cli_args=sys.argv[1:])

    print("\n" + "=" * 60)
    print("  多 LoRA 适配器合并工具")
    print("=" * 60 + "\n")

    # 读取合并配置
    method = get(cfg, "lora_merge.method", "linear")
    adapter_paths = get(cfg, "lora_merge.adapters", [])
    weights = get(cfg, "lora_merge.weights", [])
    density = get(cfg, "lora_merge.density", 0.5)
    output_path = get(cfg, "lora_merge.output_path", "merged_lora")

    # 验证合并方法
    if method not in SUPPORTED_METHODS:
        print(f"[错误] 不支持的合并方法: {method}")
        print(f"支持的方法: {list(SUPPORTED_METHODS.keys())}")
        sys.exit(1)

    # 验证适配器路径
    if not adapter_paths:
        print("[错误] 未配置待合并的适配器路径。")
        print("请在 config/config.yaml 的 lora_merge.adapters 中配置至少 2 个适配器路径。")
        sys.exit(1)

    print(f"合并配置:")
    print(f"  方法:   {method} ({SUPPORTED_METHODS[method]})")
    print(f"  适配器: {adapter_paths}")
    print(f"  权重:   {weights}")
    if method in ("ties", "dare_ties"):
        print(f"  密度:   {density}")
    print(f"  输出:   {output_path}")
    print()

    # 验证适配器兼容性（相同基础模型）
    try:
        base_model_path, adapter_configs = validate_adapters(adapter_paths)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] 适配器验证失败:\n{e}")
        sys.exit(1)

    # 验证权重
    # 如果未配置权重，默认为等权重
    if not weights:
        weights = [1.0] * len(adapter_paths)
        print(f"  未配置权重，使用等权重: {weights}")

    try:
        weights = validate_weights(weights, len(adapter_paths))
    except ValueError as e:
        print(f"[错误] 权重验证失败:\n{e}")
        sys.exit(1)

    # 检查目标模块一致性（警告级别，不阻止合并）
    all_targets = [set(c.get("target_modules", [])) for c in adapter_configs]
    if len(set(frozenset(t) for t in all_targets)) > 1:
        print("[警告] 适配器的 target_modules 不完全一致:")
        for i, (path, targets) in enumerate(zip(adapter_paths, all_targets)):
            print(f"  适配器 {i+1} ({path}): {sorted(targets)}")
        print("  PEFT 会对共有的模块进行合并，不同的模块将被忽略。")
        print()

    # 检查 rank 差异（信息级别）
    ranks = [c.get("r", 0) for c in adapter_configs]
    if len(set(ranks)) > 1:
        print(f"[提示] 适配器 rank 不同: {ranks}")
        if method == "cat":
            print(f"  cat 方法下合并后的 rank = {sum(ranks)}")
        else:
            print(f"  合并后将使用最大 rank: {max(ranks)}")
        print()

    # 执行合并
    merge_adapters(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        weights=weights,
        method=method,
        density=density,
        output_path=output_path,
        cfg=cfg,
    )

    print("=" * 60)
    print("  合并完成！")
    print("=" * 60)
    print(f"\n合并后的适配器路径: {output_path}")
    print(f"\n使用方法:")
    print(f"  # 直接推理测试")
    print(f"  python test_model.py --model {output_path}")
    print(f"\n  # 与基础模型对比测试")
    print(f"  python test_model.py --compare --model {output_path}")
    print()


if __name__ == "__main__":
    main()
