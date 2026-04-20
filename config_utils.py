"""
配置加载工具模块

功能：
1. 加载 config/config.yaml（不存在则回退到 config/config.example.yaml 并输出警告）
2. 支持 CLI 参数覆盖任意配置项（使用点号分隔的路径，如 --training.learning_rate 3e-5）
3. 提供类型自动推断（bool、int、float、str）
4. 可被 train.py、generate_data.py、test_model.py 导入使用

用法示例：
    from config_utils import load_config
    cfg = load_config()                          # 纯配置文件
    cfg = load_config(cli_args=sys.argv[1:])     # 配置文件 + CLI 覆盖
    cfg = load_config(cli_args=["--training.learning_rate", "3e-5", "--lora.r", "16"])
"""

import os
import sys
import copy
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ============================================================
# 默认配置文件路径（相对于本模块所在目录）
# ============================================================
_MODULE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _MODULE_DIR / "config"
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_CONFIG_EXAMPLE_PATH = _CONFIG_DIR / "config.example.yaml"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典。override 中的值覆盖 base 中的同名键。
    嵌套字典会递归合并，而非整体替换。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _infer_type(value_str: str) -> Any:
    """
    自动推断 CLI 参数值的类型。

    优先级：bool > None > int > float > str
    """
    # bool
    if value_str.lower() in ("true", "yes", "on"):
        return True
    if value_str.lower() in ("false", "no", "off"):
        return False

    # None
    if value_str.lower() in ("none", "null", "~"):
        return None

    # int
    try:
        # 排除科学计数法（如 3e5）被误判为 int
        if "e" not in value_str.lower() and "." not in value_str:
            return int(value_str)
    except ValueError:
        pass

    # float
    try:
        return float(value_str)
    except ValueError:
        pass

    # str
    return value_str


def _set_nested(d: Dict, dotted_key: str, value: Any) -> None:
    """
    通过点号分隔的路径设置嵌套字典的值。

    例如: _set_nested(d, "training.learning_rate", 3e-5)
    等价于: d["training"]["learning_rate"] = 3e-5

    如果中间路径不存在，会自动创建空字典。
    """
    keys = dotted_key.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_nested(d: Dict, dotted_key: str, default: Any = None) -> Any:
    """
    通过点号分隔的路径获取嵌套字典的值。

    例如: _get_nested(d, "training.learning_rate")
    等价于: d["training"]["learning_rate"]
    """
    keys = dotted_key.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_cli_overrides(args: List[str]) -> Dict:
    """
    解析 CLI 参数为嵌套字典。

    支持格式：
        --training.learning_rate 3e-5
        --lora.r 16
        --model.load_in_4bit true
        --output.merge_model false
        --lora.target_modules q_proj,k_proj,v_proj  (逗号分隔转为列表)

    返回嵌套字典，可直接用于 _deep_merge。

    不以 -- 开头的参数会被忽略（方便与 argparse 等混用）。
    """
    overrides: Dict = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # 去掉 --
            # 检查是否有值
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                raw_value = args[i + 1]
                i += 2
            else:
                # 无值的标志参数，视为 true
                raw_value = "true"
                i += 1

            # 逗号分隔的值转为列表
            if "," in raw_value and not raw_value.startswith("["):
                value = [_infer_type(v.strip()) for v in raw_value.split(",")]
            else:
                value = _infer_type(raw_value)

            _set_nested(overrides, key, value)
        else:
            i += 1

    return overrides


def load_yaml(path: Path) -> Dict:
    """加载 YAML 文件，返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
) -> Dict:
    """
    加载配置。

    优先级（从高到低）：
    1. CLI 参数覆盖
    2. config/config.yaml
    3. config/config.example.yaml（回退）

    参数：
        config_path: 指定配置文件路径。为 None 时按默认路径查找。
        cli_args: CLI 参数列表。为 None 时不解析 CLI。
                  传入 sys.argv[1:] 即可从命令行读取。

    返回：
        合并后的配置字典。
    """
    # 确定配置文件路径
    if config_path is not None:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {cfg_path}")
        config = load_yaml(cfg_path)
    elif _CONFIG_PATH.exists():
        config = load_yaml(_CONFIG_PATH)
    elif _CONFIG_EXAMPLE_PATH.exists():
        warnings.warn(
            f"config/config.yaml 不存在，已回退到 {_CONFIG_EXAMPLE_PATH.name}。"
            f"建议执行: cp config/config.example.yaml config/config.yaml",
            UserWarning,
            stacklevel=2,
        )
        config = load_yaml(_CONFIG_EXAMPLE_PATH)
    else:
        raise FileNotFoundError(
            f"找不到配置文件。请确保 {_CONFIG_PATH} 或 {_CONFIG_EXAMPLE_PATH} 存在。"
        )

    # 合并 CLI 覆盖
    if cli_args:
        overrides = parse_cli_overrides(cli_args)
        config = _deep_merge(config, overrides)

    return config


def get(config: Dict, key: str, default: Any = None) -> Any:
    """
    便捷方法：通过点号路径获取配置值。

    用法：
        cfg = load_config()
        lr = get(cfg, "training.learning_rate", 5e-5)
        model_path = get(cfg, "model.path")
    """
    return _get_nested(config, key, default)


def print_config(config: Dict, title: str = "当前配置") -> None:
    """打印配置（用于训练开始前的确认输出）。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    _print_dict(config, indent=2)
    print(f"{'=' * 60}\n")


def _print_dict(d: Dict, indent: int = 0) -> None:
    """递归打印字典。"""
    prefix = " " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_dict(value, indent + 2)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}: {value}")


# ============================================================
# 直接运行时输出当前配置（用于调试）
# ============================================================
if __name__ == "__main__":
    cfg = load_config(cli_args=sys.argv[1:])
    print_config(cfg)
