"""
数据生成脚本：为 QLoRA 微调生成训练数据

功能：
1. 从 data/seeds/ 加载种子样本（手写高质量样本）
2. 调用外部 LLM API 批量生成训练数据变体
3. 自动格式验证（JSON结构、messages 字段、tool_call 格式）
4. 多文件批次管理，支持断点续传
5. 去重合并，按类别比例采样输出 training_data.jsonl
6. 数据质量审查：自动检查 + 人工审查 + 统计报告

用法：
    # 生成 2000 条 java_coding 类别数据
    python generate_data.py --count 2000 --category java_coding

    # 生成所有类别数据（按 config 中的比例分配）
    python generate_data.py --count 2000

    # 合并 seeds + generated 数据，去重、按比例采样、输出 training_data.jsonl
    python generate_data.py --merge

    # 自动审查并生成数据质量报告 data_report.json
    python generate_data.py --report

    # 交互式人工审查模式（逐条 accept/reject/edit）
    python generate_data.py --review

    # 使用自定义配置
    python generate_data.py --count 500 --data.llm_api.model deepseek-chat

数据格式：JSONL，每行一个 {"messages": [...]} 对象
tool_call 使用 XML 格式：
    <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
"""

import json
import hashlib
import math
import os
import re
import sys
import time
import random
import logging
import yaml
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 配置加载
# ============================================================
from config_utils import load_config, get

# ============================================================
# 常量定义
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# 0. 动态类型定义加载
# ============================================================

def load_categories(categories_dir: str = "data/categories") -> Dict[str, dict]:
    """
    动态扫描并加载 data/categories/ 下的所有 YAML 类型定义文件。

    每个 YAML 文件必须包含以下必填字段：
    - name: 类型名称
    - subcategories: 子话题列表
    - meta_prompt: 用于生成训练数据的 prompt 模板

    可选字段：
    - description: 类型描述
    - extra_instructions: 子话题到详细指令的映射 dict

    参数:
        categories_dir: 类型定义目录的相对路径（相对于脚本目录）

    返回:
        Dict[str, dict] — key 为 YAML 文件名 stem（如 "java_coding"），value 为解析后的 dict
    """
    categories_path = SCRIPT_DIR / categories_dir
    result: Dict[str, dict] = {}
    required_fields = ["name", "subcategories", "meta_prompt"]

    if not categories_path.exists():
        logger.error("类型定义目录不存在: %s", categories_path)
        return result

    if not categories_path.is_dir():
        logger.error("类型定义路径不是目录: %s", categories_path)
        return result

    yaml_files = sorted(categories_path.glob("*.yaml"))
    if not yaml_files:
        logger.warning("类型定义目录为空: %s", categories_path)
        return result

    for yaml_file in yaml_files:
        category_name = yaml_file.stem  # e.g. "java_coding"
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("YAML 解析失败 [%s]: %s", yaml_file.name, e)
            continue

        if not isinstance(data, dict):
            logger.error("YAML 文件内容不是 dict [%s]", yaml_file.name)
            continue

        # 校验必填字段
        missing = [field for field in required_fields if field not in data]
        if missing:
            logger.error(
                "类型定义缺少必填字段 [%s]: %s", yaml_file.name, ", ".join(missing)
            )
            continue

        # 校验 subcategories 是列表
        if not isinstance(data["subcategories"], list) or len(data["subcategories"]) == 0:
            logger.error("subcategories 必须是非空列表 [%s]", yaml_file.name)
            continue

        # 校验 meta_prompt 是字符串
        if not isinstance(data["meta_prompt"], str):
            logger.error("meta_prompt 必须是字符串 [%s]", yaml_file.name)
            continue

        # 确保 extra_instructions 存在（可选字段，默认空 dict）
        if "extra_instructions" not in data:
            data["extra_instructions"] = {}

        result[category_name] = data

    # 日志输出发现的类型列表
    if result:
        type_summary = ", ".join(
            f"{name}({len(d['subcategories'])} subcategories)"
            for name, d in sorted(result.items())
        )
        logger.info("已加载 %d 个样本类型定义: %s", len(result), type_summary)
    else:
        logger.warning("未发现任何有效的样本类型定义")

    return result


# ============================================================
# 1. 种子样本加载
# ============================================================

def load_seeds(seeds_dir: str = "data/seeds") -> Dict[str, List[Dict]]:
    """
    从 data/seeds/ 目录加载种子样本。

    返回格式：
        {
            "java_coding": [{"messages": [...], "metadata": {...}}, ...],
            "tool_calling": [...],
            "frontend_dev": [...]
        }
    """
    seeds_path = SCRIPT_DIR / seeds_dir
    result: Dict[str, List[Dict]] = {}

    if not seeds_path.exists():
        logger.warning("种子目录不存在: %s", seeds_path)
        return result

    for jsonl_file in sorted(seeds_path.glob("*.jsonl")):
        category = jsonl_file.stem  # e.g. "java_coding"
        samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "种子文件 %s 第 %d 行 JSON 解析失败: %s",
                        jsonl_file.name, line_num, e,
                    )
        result[category] = samples
        logger.info("加载种子文件 %s: %d 条样本", jsonl_file.name, len(samples))

    return result


# ============================================================
# 2. 格式验证
# ============================================================

def validate_sample(sample: Dict, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    验证单个训练样本的格式。

    检查项：
    - JSON 结构完整（messages 字段存在且非空）
    - 每条 message 包含 role 和 content
    - role 值合法（system, user, assistant）
    - 对话至少包含 user + assistant 各一条
    - tool_call XML 格式正确（如果存在）

    返回：(是否通过, 错误信息列表)
    """
    errors: List[str] = []

    # 基础结构
    if not isinstance(sample, dict):
        return False, ["样本不是字典类型"]

    if "messages" not in sample:
        return False, ["缺少 'messages' 字段"]

    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False, ["'messages' 字段应为至少包含2条消息的列表"]

    # 消息格式检查
    valid_roles = {"system", "user", "assistant"}
    has_user = False
    has_assistant = False

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"第 {i+1} 条消息不是字典类型")
            continue

        if "role" not in msg:
            errors.append(f"第 {i+1} 条消息缺少 'role' 字段")
            continue

        if msg["role"] not in valid_roles:
            errors.append(f"第 {i+1} 条消息 role 值无效: {msg['role']}")

        if "content" not in msg:
            errors.append(f"第 {i+1} 条消息缺少 'content' 字段")
            continue

        if not isinstance(msg["content"], str):
            errors.append(f"第 {i+1} 条消息 content 不是字符串")
            continue

        if msg["role"] == "user":
            has_user = True
        elif msg["role"] == "assistant":
            has_assistant = True

        # tool_call 格式检查
        if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
            tc_errors = _validate_tool_calls(msg["content"], i + 1)
            errors.extend(tc_errors)

    if not has_user:
        errors.append("对话中缺少 user 消息")
    if not has_assistant:
        errors.append("对话中缺少 assistant 消息")

    # 异常长度检测
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
            content_len = len(msg["content"])
            if content_len < 2:
                errors.append(f"第 {i+1} 条消息内容过短 ({content_len} 字符)")
            if content_len > 20000:
                errors.append(f"第 {i+1} 条消息内容过长 ({content_len} 字符)")

    return len(errors) == 0, errors


def _validate_tool_calls(content: str, msg_index: int) -> List[str]:
    """验证 assistant 消息中的 tool_call XML 格式。"""
    errors: List[str] = []

    # 检查 <tool_call> 和 </tool_call> 配对
    open_count = content.count("<tool_call>")
    close_count = content.count("</tool_call>")
    if open_count != close_count:
        errors.append(
            f"第 {msg_index} 条消息: <tool_call> 标签未配对 "
            f"(开 {open_count}, 闭 {close_count})"
        )
        return errors

    # 提取每个 tool_call 块并验证内部结构
    tc_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for i, match in enumerate(tc_pattern.finditer(content)):
        tc_content = match.group(1)

        # 检查 <function=name> 存在
        func_match = re.search(r"<function=(\w+)>", tc_content)
        if not func_match:
            errors.append(
                f"第 {msg_index} 条消息第 {i+1} 个 tool_call: "
                f"缺少 <function=name> 标签"
            )
            continue

        # 检查 </function> 关闭
        if "</function>" not in tc_content:
            errors.append(
                f"第 {msg_index} 条消息第 {i+1} 个 tool_call: "
                f"缺少 </function> 关闭标签"
            )

        # 检查 parameter 配对
        param_open = len(re.findall(r"<parameter=\w+>", tc_content))
        param_close = tc_content.count("</parameter>")
        if param_open != param_close:
            errors.append(
                f"第 {msg_index} 条消息第 {i+1} 个 tool_call: "
                f"<parameter> 标签未配对 (开 {param_open}, 闭 {param_close})"
            )

    return errors


def validate_tool_response(content: str) -> Tuple[bool, List[str]]:
    """验证 tool_response 格式。"""
    errors: List[str] = []
    if "<tool_response>" in content:
        open_count = content.count("<tool_response>")
        close_count = content.count("</tool_response>")
        if open_count != close_count:
            errors.append(
                f"<tool_response> 标签未配对 (开 {open_count}, 闭 {close_count})"
            )
    return len(errors) == 0, errors


def validate_batch(samples: List[Dict]) -> Tuple[int, int, List[str]]:
    """
    批量验证样本。

    返回：(通过数, 失败数, 所有错误信息)
    """
    passed = 0
    failed = 0
    all_errors: List[str] = []

    for i, sample in enumerate(samples):
        valid, errors = validate_sample(sample)
        if valid:
            passed += 1
        else:
            failed += 1
            for e in errors:
                all_errors.append(f"样本 {i+1}: {e}")

    return passed, failed, all_errors


# ============================================================
# 3. 去重与相似度检测
# ============================================================

def _extract_text(sample: Dict) -> str:
    """从样本的 messages 中提取所有文本内容，用于相似度计算。"""
    parts = []
    for msg in sample.get("messages", []):
        content = msg.get("content", "")
        if content:
            parts.append(content)
    return "\n".join(parts)


def _content_hash(sample: Dict) -> str:
    """计算样本内容的 MD5 哈希值（精确去重）。"""
    text = _extract_text(sample)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """提取字符级 n-gram 集合。"""
    text = text.lower().replace(" ", "").replace("\n", "")
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """计算两个集合的 Jaccard 相似度。"""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_samples(
    samples: List[Dict],
    similarity_threshold: float = 0.85,
    ngram_size: int = 3,
) -> Tuple[List[Dict], int]:
    """
    对样本列表进行去重。

    两轮去重：
    1. 精确去重：MD5 哈希值完全相同的样本只保留一个
    2. 相似去重：字符级 n-gram Jaccard 相似度超过阈值的样本只保留一个

    参数：
        samples: 样本列表
        similarity_threshold: 相似度阈值（0.0-1.0），超过此值视为重复
        ngram_size: n-gram 大小

    返回：
        (去重后的样本列表, 去除的重复数)
    """
    if not samples:
        return [], 0

    original_count = len(samples)

    # 第一轮：精确去重（MD5 哈希）
    seen_hashes: Set[str] = set()
    unique_samples: List[Dict] = []
    for sample in samples:
        h = _content_hash(sample)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_samples.append(sample)

    hash_dupes = original_count - len(unique_samples)
    if hash_dupes > 0:
        logger.info("精确去重: 移除 %d 条完全相同的样本", hash_dupes)

    # 第二轮：相似去重（n-gram Jaccard）
    # 为避免 O(n^2) 对大量数据过慢，只在样本量可控时执行
    if len(unique_samples) <= 5000:
        ngram_cache: List[Tuple[Dict, Set[str]]] = []
        deduped: List[Dict] = []

        for sample in unique_samples:
            text = _extract_text(sample)
            ngrams = _char_ngrams(text, ngram_size)

            is_duplicate = False
            for _, existing_ngrams in ngram_cache:
                sim = _jaccard_similarity(ngrams, existing_ngrams)
                if sim >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append(sample)
                ngram_cache.append((sample, ngrams))

        sim_dupes = len(unique_samples) - len(deduped)
        if sim_dupes > 0:
            logger.info(
                "相似去重 (阈值=%.2f): 移除 %d 条相似样本",
                similarity_threshold, sim_dupes,
            )
        unique_samples = deduped
    else:
        logger.info("样本数 > 5000，跳过相似去重（仅执行精确去重）")

    total_removed = original_count - len(unique_samples)
    return unique_samples, total_removed


# ============================================================
# 3.5 批次文件管理
# ============================================================

def get_next_batch_number(generated_dir: Path, category: str) -> int:
    """
    扫描 data/generated/ 目录，获取指定类别的下一个批次号。

    文件命名格式：{category}_{batch_number:03d}.jsonl
    例如：java_coding_001.jsonl, java_coding_002.jsonl
    """
    if not generated_dir.exists():
        return 1

    pattern = re.compile(rf"^{re.escape(category)}_(\d+)\.jsonl$")
    max_num = 0
    for f in generated_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num

    return max_num + 1


def count_existing_samples(generated_dir: Path, category: Optional[str] = None) -> int:
    """
    统计 data/generated/ 目录下已有样本数量。

    参数：
        generated_dir: 生成数据目录
        category: 如果指定，只统计该类别的样本数；None 则统计所有
    """
    if not generated_dir.exists():
        return 0

    count = 0
    for f in sorted(generated_dir.glob("*.jsonl")):
        if category:
            # 只统计匹配类别的文件
            if not f.name.startswith(f"{category}_"):
                continue
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    return count


def load_all_generated(generated_dir: Path, category: Optional[str] = None) -> List[Dict]:
    """
    加载 data/generated/ 目录下所有生成的样本。

    参数：
        generated_dir: 生成数据目录
        category: 如果指定，只加载该类别；None 则加载所有
    """
    samples: List[Dict] = []
    if not generated_dir.exists():
        return samples

    for f in sorted(generated_dir.glob("*.jsonl")):
        if category and not f.name.startswith(f"{category}_"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning("文件 %s 第 %d 行 JSON 解析失败: %s", f.name, line_num, e)

    return samples


def merge_all_data(
    seeds_dir: Path,
    generated_dir: Path,
    output_path: Path,
    category_ratios: Dict[str, float],
    target_size: Optional[int] = None,
    similarity_threshold: float = 0.85,
    seed: int = 42,
) -> int:
    """
    合并 seeds + generated 数据，去重、按比例采样、打散、输出 training_data.jsonl。

    流程：
    1. 加载所有种子样本（data/seeds/）
    2. 加载所有生成样本（data/generated/）
    3. 按类别分组
    4. 每个类别内去重
    5. 按配置比例采样（如果 target_size 指定）
    6. 打散输出

    参数：
        seeds_dir: 种子样本目录
        generated_dir: 生成数据目录
        output_path: 输出文件路径
        category_ratios: 各类别比例
        target_size: 目标总样本数（None 则使用全部去重后的样本）
        similarity_threshold: 去重相似度阈值
        seed: 随机种子

    返回：
        最终输出的样本数
    """
    random.seed(seed)

    # 1. 按类别收集所有样本
    category_samples: Dict[str, List[Dict]] = defaultdict(list)

    # 加载种子样本
    seed_count = 0
    if seeds_dir.exists():
        for f in sorted(seeds_dir.glob("*.jsonl")):
            category = f.stem
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        # 确保有 metadata
                        if "metadata" not in sample:
                            sample["metadata"] = {"category": category, "source": "seed"}
                        category_samples[category].append(sample)
                        seed_count += 1
                    except json.JSONDecodeError:
                        pass
    logger.info("加载种子样本: %d 条", seed_count)

    # 加载生成样本
    generated_samples = load_all_generated(generated_dir)
    gen_count = len(generated_samples)
    for sample in generated_samples:
        cat = sample.get("metadata", {}).get("category", "unknown")
        category_samples[cat].append(sample)
    logger.info("加载生成样本: %d 条", gen_count)

    # 2. 每个类别内去重
    total_before = 0
    total_after = 0
    deduped_by_category: Dict[str, List[Dict]] = {}

    for cat, samples in category_samples.items():
        total_before += len(samples)
        deduped, removed = deduplicate_samples(
            samples, similarity_threshold=similarity_threshold
        )
        deduped_by_category[cat] = deduped
        total_after += len(deduped)
        logger.info(
            "类别 '%s': %d 条 → 去重后 %d 条 (移除 %d)",
            cat, len(samples), len(deduped), removed,
        )

    logger.info("去重总计: %d → %d (移除 %d)", total_before, total_after, total_before - total_after)

    # 3. 按比例采样
    final_samples: List[Dict] = []

    if target_size and target_size < total_after:
        # 按比例从各类别采样
        for cat, ratio in category_ratios.items():
            cat_samples = deduped_by_category.get(cat, [])
            cat_target = max(1, int(target_size * ratio))
            if len(cat_samples) <= cat_target:
                final_samples.extend(cat_samples)
            else:
                random.shuffle(cat_samples)
                final_samples.extend(cat_samples[:cat_target])
            logger.info(
                "采样类别 '%s': %d/%d 条 (比例 %.0f%%)",
                cat, min(len(cat_samples), cat_target), len(cat_samples), ratio * 100,
            )

        # 包含没有在 ratios 中定义的类别（全部保留）
        for cat, samples in deduped_by_category.items():
            if cat not in category_ratios:
                final_samples.extend(samples)
                logger.info("类别 '%s' (未配置比例): 全部保留 %d 条", cat, len(samples))
    else:
        # 不限制数量，使用全部去重后样本
        for cat, samples in deduped_by_category.items():
            final_samples.extend(samples)

    # 4. 打散
    random.shuffle(final_samples)

    # 5. 输出（保留 metadata.category，去掉其他 metadata 字段）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_samples:
            out = {"messages": sample["messages"]}
            # 保留 category 信息用于 train.py 分层采样
            category = sample.get("metadata", {}).get("category", "")
            if category:
                out["metadata"] = {"category": category}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info("合并完成:")
    logger.info("  种子样本: %d 条", seed_count)
    logger.info("  生成样本: %d 条", gen_count)
    logger.info("  去重后: %d 条", total_after)
    logger.info("  最终输出: %d 条 → %s", len(final_samples), output_path)

    # 输出各类别统计
    cat_counts: Counter = Counter()
    for sample in final_samples:
        cat = sample.get("metadata", {}).get("category", "unknown")
        cat_counts[cat] += 1
    for cat, cnt in cat_counts.most_common():
        logger.info("    %s: %d 条 (%.1f%%)", cat, cnt, cnt / len(final_samples) * 100 if final_samples else 0)

    logger.info("=" * 60)
    return len(final_samples)


# ============================================================
# 3.6 数据质量审查与报告
# ============================================================

def review_sample_detailed(sample: Dict, index: int = 0) -> Tuple[bool, List[str], List[str]]:
    """
    对单个样本执行详细的质量审查。

    除了基础格式验证外，还检查：
    - tool_call XML 格式正确性
    - tool_response 与 tool_call 配对
    - 消息轮次完整性（不能连续两条 user 或 assistant）
    - 异常回复长度（过短或过长）

    返回：(是否通过, 错误列表, 警告列表)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 基础格式验证（复用 validate_sample）
    valid, base_errors = validate_sample(sample)
    errors.extend(base_errors)

    if not isinstance(sample, dict) or "messages" not in sample:
        return False, errors, warnings

    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return False, errors, warnings

    # 检查 tool_response 格式
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        # 检查 tool_response 标签配对
        if "<tool_response>" in content or "</tool_response>" in content:
            tr_valid, tr_errors = validate_tool_response(content)
            if not tr_valid:
                for e in tr_errors:
                    errors.append(f"第 {i+1} 条消息: {e}")

    # 检查消息轮次完整性（连续相同 role 检测）
    prev_role = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role == prev_role and role in ("user", "assistant"):
            # 允许 user 后跟 user（tool_response 场景），但发出警告
            if role == "user":
                content = msg.get("content", "")
                if "<tool_response>" in content:
                    pass  # tool_response 作为 user 消息是正常的
                else:
                    warnings.append(f"第 {i} 和第 {i+1} 条消息: 连续出现 {role} 消息")
            else:
                warnings.append(f"第 {i} 和第 {i+1} 条消息: 连续出现 {role} 消息")
        prev_role = role

    # 检查 tool_call 与 tool_response 配对
    tc_count = 0
    tr_count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        tc_count += content.count("<tool_call>")
        tr_count += content.count("<tool_response>")

    if tc_count > 0 and tr_count == 0:
        warnings.append(f"有 {tc_count} 个 tool_call 但没有 tool_response")
    if tr_count > 0 and tc_count == 0:
        warnings.append(f"有 {tr_count} 个 tool_response 但没有 tool_call")
    if tc_count > 0 and tr_count > 0 and tc_count != tr_count:
        warnings.append(
            f"tool_call ({tc_count}) 与 tool_response ({tr_count}) 数量不匹配"
        )

    # 检查 assistant 回复的平均长度
    assistant_lengths = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                assistant_lengths.append(len(content))

    if assistant_lengths:
        avg_len = sum(assistant_lengths) / len(assistant_lengths)
        if avg_len < 10:
            errors.append(f"assistant 回复平均长度过短 ({avg_len:.0f} 字符)")
        elif avg_len < 30:
            warnings.append(f"assistant 回复平均长度偏短 ({avg_len:.0f} 字符)")
        if avg_len > 15000:
            warnings.append(f"assistant 回复平均长度偏长 ({avg_len:.0f} 字符)")

    passed = len(errors) == 0
    return passed, errors, warnings


def review_all_data(
    seeds_dir: Path,
    generated_dir: Path,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    对所有数据（seeds + generated）执行自动质量审查。

    返回：(通过的样本列表, 失败的样本列表, 带警告的样本列表)
    每个样本会增加 _review 字段记录审查结果。
    """
    all_samples: List[Dict] = []

    # 加载种子样本
    if seeds_dir.exists():
        for f in sorted(seeds_dir.glob("*.jsonl")):
            category = f.stem
            with open(f, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        if "metadata" not in sample:
                            sample["metadata"] = {"category": category, "source": "seed"}
                        sample["_source_file"] = f.name
                        sample["_source_line"] = line_num
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        pass

    # 加载生成样本
    if generated_dir.exists():
        for f in sorted(generated_dir.glob("*.jsonl")):
            with open(f, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        sample["_source_file"] = f.name
                        sample["_source_line"] = line_num
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        pass

    passed_samples: List[Dict] = []
    failed_samples: List[Dict] = []
    warned_samples: List[Dict] = []

    for i, sample in enumerate(all_samples):
        passed, errors, warnings = review_sample_detailed(sample, i)
        sample["_review"] = {
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
        }

        if passed:
            passed_samples.append(sample)
            if warnings:
                warned_samples.append(sample)
        else:
            failed_samples.append(sample)

    logger.info("=" * 60)
    logger.info("自动质量审查结果:")
    logger.info("  总样本数: %d", len(all_samples))
    logger.info("  通过: %d (%.1f%%)", len(passed_samples),
                len(passed_samples) / len(all_samples) * 100 if all_samples else 0)
    logger.info("  失败: %d (%.1f%%)", len(failed_samples),
                len(failed_samples) / len(all_samples) * 100 if all_samples else 0)
    logger.info("  有警告: %d", len(warned_samples))
    logger.info("=" * 60)

    return passed_samples, failed_samples, warned_samples


def generate_data_report(
    seeds_dir: Path,
    generated_dir: Path,
    output_path: Path,
) -> Dict:
    """
    生成数据质量报告 data_report.json。

    报告内容：
    - 总样本数
    - 类别分布
    - 平均轮次数
    - 平均 token 数（字符数近似）
    - 格式错误数
    - 每个类别的详细统计

    返回报告字典。
    """
    passed, failed, warned = review_all_data(seeds_dir, generated_dir)

    all_samples = passed + failed

    # 类别分布
    category_dist: Dict[str, int] = Counter()
    source_dist: Dict[str, int] = Counter()  # seed vs generated

    # 统计指标
    turn_counts: List[int] = []
    char_counts: List[int] = []
    assistant_char_counts: List[int] = []

    # 每个类别的详细统计
    category_details: Dict[str, Dict] = {}

    for sample in all_samples:
        cat = sample.get("metadata", {}).get("category", "unknown")
        source = sample.get("metadata", {}).get("source", "unknown")
        category_dist[cat] += 1
        source_dist[source] += 1

        messages = sample.get("messages", [])
        turn_count = len(messages)
        turn_counts.append(turn_count)

        total_chars = 0
        assistant_chars = 0
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                    if msg.get("role") == "assistant":
                        assistant_chars += len(content)

        char_counts.append(total_chars)
        assistant_char_counts.append(assistant_chars)

        # 累计到类别详情
        if cat not in category_details:
            category_details[cat] = {
                "count": 0,
                "passed": 0,
                "failed": 0,
                "warned": 0,
                "turn_counts": [],
                "char_counts": [],
            }
        cd = category_details[cat]
        cd["count"] += 1
        cd["turn_counts"].append(turn_count)
        cd["char_counts"].append(total_chars)

        review = sample.get("_review", {})
        if review.get("passed", True):
            cd["passed"] += 1
            if review.get("warnings"):
                cd["warned"] += 1
        else:
            cd["failed"] += 1

    # 构建报告
    report: Dict[str, Any] = {
        "summary": {
            "total_samples": len(all_samples),
            "passed_samples": len(passed),
            "failed_samples": len(failed),
            "warned_samples": len(warned),
            "pass_rate": round(len(passed) / len(all_samples) * 100, 1) if all_samples else 0,
        },
        "category_distribution": dict(category_dist.most_common()),
        "source_distribution": dict(source_dist.most_common()),
        "statistics": {
            "avg_turn_count": round(sum(turn_counts) / len(turn_counts), 1) if turn_counts else 0,
            "min_turn_count": min(turn_counts) if turn_counts else 0,
            "max_turn_count": max(turn_counts) if turn_counts else 0,
            "avg_char_count": round(sum(char_counts) / len(char_counts), 0) if char_counts else 0,
            "min_char_count": min(char_counts) if char_counts else 0,
            "max_char_count": max(char_counts) if char_counts else 0,
            "avg_assistant_char_count": round(
                sum(assistant_char_counts) / len(assistant_char_counts), 0
            ) if assistant_char_counts else 0,
        },
        "category_details": {},
        "errors": [],
    }

    # 每个类别的聚合统计
    for cat, cd in sorted(category_details.items()):
        report["category_details"][cat] = {
            "count": cd["count"],
            "passed": cd["passed"],
            "failed": cd["failed"],
            "warned": cd["warned"],
            "avg_turn_count": round(sum(cd["turn_counts"]) / len(cd["turn_counts"]), 1) if cd["turn_counts"] else 0,
            "avg_char_count": round(sum(cd["char_counts"]) / len(cd["char_counts"]), 0) if cd["char_counts"] else 0,
        }

    # 收集失败样本的错误信息
    for sample in failed:
        review = sample.get("_review", {})
        report["errors"].append({
            "source_file": sample.get("_source_file", "unknown"),
            "source_line": sample.get("_source_line", 0),
            "category": sample.get("metadata", {}).get("category", "unknown"),
            "errors": review.get("errors", []),
        })

    # 写入报告文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("数据质量报告已写入: %s", output_path)

    # 打印摘要
    logger.info("=" * 60)
    logger.info("数据质量报告摘要:")
    logger.info("  总样本数: %d", report["summary"]["total_samples"])
    logger.info("  通过率: %.1f%% (%d/%d)",
                report["summary"]["pass_rate"],
                report["summary"]["passed_samples"],
                report["summary"]["total_samples"])
    logger.info("  失败: %d, 有警告: %d",
                report["summary"]["failed_samples"],
                report["summary"]["warned_samples"])
    logger.info("  平均轮次: %.1f, 平均字符数: %.0f",
                report["statistics"]["avg_turn_count"],
                report["statistics"]["avg_char_count"])
    logger.info("  类别分布: %s",
                ", ".join(f"{k}={v}" for k, v in report["category_distribution"].items()))
    logger.info("=" * 60)

    return report


def interactive_review(
    seeds_dir: Path,
    generated_dir: Path,
    rejected_path: Path,
) -> Tuple[int, int, int]:
    """
    交互式人工审查模式。

    逐条显示样本，支持操作：
    - [a]ccept: 接受样本
    - [r]eject: 拒绝样本（需输入原因）
    - [e]dit: 编辑样本（打开临时文件编辑 JSON）
    - [s]kip: 跳过当前样本（保持原状）
    - [q]uit: 退出审查

    被拒绝的样本记录到 rejected_samples.jsonl。

    返回：(accepted_count, rejected_count, skipped_count)
    """
    # 先执行自动审查
    passed, failed, warned = review_all_data(seeds_dir, generated_dir)

    # 只审查需要关注的样本：失败的 + 有警告的
    # 完全通过且无警告的不需要人工审查
    review_queue: List[Dict] = []

    # 失败的样本优先审查
    for sample in failed:
        sample["_review_priority"] = "FAILED"
        review_queue.append(sample)

    # 有警告的样本其次
    for sample in warned:
        if sample not in failed:  # 避免重复
            sample["_review_priority"] = "WARNING"
            review_queue.append(sample)

    if not review_queue:
        logger.info("所有样本均通过自动审查，无需人工审查")
        return len(passed), 0, 0

    logger.info(
        "需要人工审查的样本: %d 条 (失败 %d + 有警告 %d)",
        len(review_queue), len(failed), len(warned) - len([s for s in warned if s in failed]),
    )

    accepted_count = len(passed) - len(warned)  # 完全通过的
    rejected_count = 0
    skipped_count = 0
    edited_count = 0

    rejected_samples: List[Dict] = []

    print("\n" + "=" * 60)
    print("交互式人工审查模式")
    print("操作: [a]ccept  [r]eject  [e]dit  [s]kip  [q]uit")
    print("=" * 60)

    for idx, sample in enumerate(review_queue):
        priority = sample.get("_review_priority", "UNKNOWN")
        review = sample.get("_review", {})
        source_file = sample.get("_source_file", "unknown")
        source_line = sample.get("_source_line", 0)
        category = sample.get("metadata", {}).get("category", "unknown")
        subcategory = sample.get("metadata", {}).get("subcategory", "")

        print(f"\n--- 样本 {idx+1}/{len(review_queue)} [{priority}] ---")
        print(f"来源: {source_file}:{source_line}  类别: {category}/{subcategory}")

        if review.get("errors"):
            print(f"错误: {'; '.join(review['errors'])}")
        if review.get("warnings"):
            print(f"警告: {'; '.join(review['warnings'])}")

        # 显示消息摘要
        messages = sample.get("messages", [])
        print(f"消息数: {len(messages)}")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                # 显示前 200 字符
                preview = content[:200].replace("\n", "\\n")
                if len(content) > 200:
                    preview += "..."
                print(f"  [{role}] ({len(content)} chars): {preview}")
            else:
                print(f"  [{role}] (非字符串 content)")

        # 获取用户操作
        while True:
            try:
                action = input("\n操作 [a/r/e/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n中断，保存已有结果...")
                action = "q"

            if action in ("a", "accept"):
                accepted_count += 1
                print("  → 已接受")
                break
            elif action in ("r", "reject"):
                reason = ""
                try:
                    reason = input("拒绝原因: ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = "用户中断"
                if not reason:
                    reason = "人工审查拒绝"
                rejected_count += 1

                # 记录拒绝信息
                rejected_record = {
                    "messages": sample.get("messages", []),
                    "metadata": sample.get("metadata", {}),
                    "rejection": {
                        "reason": reason,
                        "source_file": source_file,
                        "source_line": source_line,
                        "auto_errors": review.get("errors", []),
                        "auto_warnings": review.get("warnings", []),
                    },
                }
                rejected_samples.append(rejected_record)
                print(f"  → 已拒绝: {reason}")
                break
            elif action in ("e", "edit"):
                # 编辑模式：写入临时文件让用户编辑
                import tempfile
                tmp_path = SCRIPT_DIR / ".review_edit_tmp.json"
                sample_to_edit = {
                    "messages": sample.get("messages", []),
                    "metadata": sample.get("metadata", {}),
                }
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(sample_to_edit, f, ensure_ascii=False, indent=2)

                editor = os.environ.get("EDITOR", "vim")
                print(f"  请编辑文件: {tmp_path}")
                print(f"  将使用编辑器: {editor}")
                os.system(f'{editor} "{tmp_path}"')

                # 读取编辑后的内容
                try:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        edited_sample = json.load(f)
                    # 验证编辑后的样本
                    valid, errors = validate_sample(edited_sample)
                    if valid:
                        sample["messages"] = edited_sample["messages"]
                        if "metadata" in edited_sample:
                            sample["metadata"] = edited_sample["metadata"]
                        accepted_count += 1
                        edited_count += 1
                        print("  → 已编辑并接受")
                    else:
                        print(f"  编辑后的样本验证失败: {'; '.join(errors)}")
                        print("  请重新选择操作")
                        continue
                except (json.JSONDecodeError, Exception) as e:
                    print(f"  读取编辑结果失败: {e}")
                    print("  请重新选择操作")
                    continue
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                break
            elif action in ("s", "skip"):
                skipped_count += 1
                print("  → 已跳过")
                break
            elif action in ("q", "quit"):
                skipped_count += len(review_queue) - idx
                print(f"  退出审查。剩余 {len(review_queue) - idx} 条样本已跳过。")
                break
            else:
                print("  无效操作，请输入 a/r/e/s/q")

        if action in ("q", "quit"):
            break

    # 写入被拒绝的样本
    if rejected_samples:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        # 追加模式
        with open(rejected_path, "a", encoding="utf-8") as f:
            for sample in rejected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("已追加 %d 条被拒绝样本到 %s", len(rejected_samples), rejected_path)

    print("\n" + "=" * 60)
    print("人工审查完成:")
    print(f"  接受: {accepted_count} (其中编辑后接受: {edited_count})")
    print(f"  拒绝: {rejected_count}")
    print(f"  跳过: {skipped_count}")
    print("=" * 60)

    return accepted_count, rejected_count, skipped_count


def merge_reviewed_data(
    seeds_dir: Path,
    generated_dir: Path,
    rejected_path: Path,
    output_path: Path,
    category_ratios: Dict[str, float],
    target_size: Optional[int] = None,
    similarity_threshold: float = 0.85,
    seed: int = 42,
) -> int:
    """
    合并数据，排除被拒绝的样本。

    在 merge_all_data 的基础上，额外过滤掉 rejected_samples.jsonl 中记录的样本。

    返回最终输出的样本数。
    """
    # 加载被拒绝的样本哈希集合
    rejected_hashes: Set[str] = set()
    if rejected_path.exists():
        with open(rejected_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    h = _content_hash(sample)
                    rejected_hashes.add(h)
                except (json.JSONDecodeError, Exception):
                    pass
        if rejected_hashes:
            logger.info("加载 %d 条被拒绝样本的指纹（合并时将排除）", len(rejected_hashes))

    # 执行自动审查
    passed, failed, warned = review_all_data(seeds_dir, generated_dir)

    # 只使用通过自动审查的样本
    reviewed_samples = passed

    # 进一步排除被人工拒绝的样本
    if rejected_hashes:
        before = len(reviewed_samples)
        reviewed_samples = [
            s for s in reviewed_samples
            if _content_hash(s) not in rejected_hashes
        ]
        excluded = before - len(reviewed_samples)
        if excluded > 0:
            logger.info("排除 %d 条人工拒绝的样本", excluded)

    # 按类别分组
    category_samples: Dict[str, List[Dict]] = defaultdict(list)
    for sample in reviewed_samples:
        cat = sample.get("metadata", {}).get("category", "unknown")
        category_samples[cat].append(sample)

    # 去重
    total_before = 0
    total_after = 0
    deduped_by_category: Dict[str, List[Dict]] = {}

    for cat, samples in category_samples.items():
        total_before += len(samples)
        deduped, removed = deduplicate_samples(
            samples, similarity_threshold=similarity_threshold
        )
        deduped_by_category[cat] = deduped
        total_after += len(deduped)
        logger.info(
            "类别 '%s': %d 条 → 去重后 %d 条 (移除 %d)",
            cat, len(samples), len(deduped), removed,
        )

    # 按比例采样
    random.seed(seed)
    final_samples: List[Dict] = []

    if target_size and target_size < total_after:
        for cat, ratio in category_ratios.items():
            cat_samples = deduped_by_category.get(cat, [])
            cat_target = max(1, int(target_size * ratio))
            if len(cat_samples) <= cat_target:
                final_samples.extend(cat_samples)
            else:
                random.shuffle(cat_samples)
                final_samples.extend(cat_samples[:cat_target])

        for cat, samples in deduped_by_category.items():
            if cat not in category_ratios:
                final_samples.extend(samples)
    else:
        for cat, samples in deduped_by_category.items():
            final_samples.extend(samples)

    # 打散
    random.shuffle(final_samples)

    # 输出（保留 metadata.category，去掉其他 metadata 字段和 _review 等临时字段）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_samples:
            out = {"messages": sample["messages"]}
            # 保留 category 信息用于 train.py 分层采样
            category = sample.get("metadata", {}).get("category", "")
            if category:
                out["metadata"] = {"category": category}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info("审查后合并完成: %d 条样本 → %s", len(final_samples), output_path)
    return len(final_samples)


# ============================================================
# 4. Meta-Prompt 定义（LLM 数据生成）
# ============================================================

TOOL_DEFINITIONS_TEXT = """可用工具列表：
1. read_file(path: str) - 读取指定路径的文件内容
2. write_file(path: str, content: str) - 将内容写入指定路径的文件
3. edit_file(path: str, old_text: str, new_text: str) - 通过精确字符串匹配替换来编辑文件
4. search_files(pattern: str, path?: str, include?: str) - 在指定目录中搜索匹配模式的文件内容
5. run_command(command: str, workdir?: str) - 在指定工作目录中执行shell命令
6. search_web(query: str) - 搜索网页获取信息

工具调用格式（CoPaw XML格式）：
<tool_call>
<function=工具名>
<parameter=参数名>
参数值
</parameter>
</function>
</tool_call>

工具响应格式：
<tool_response>
响应内容
</tool_response>"""

# DIFFICULTY_LEVELS 和 DIFFICULTY_INSTRUCTIONS 是跨类型共享的全局配置，保留在代码中
DIFFICULTY_LEVELS = ["basic", "intermediate", "advanced"]

DIFFICULTY_INSTRUCTIONS = {
    "basic": "基础难度：涉及单个概念或简单操作，适合初学者。",
    "intermediate": "中等难度：涉及多个概念组合或实际应用场景，需要一定的经验。",
    "advanced": "高级难度：涉及复杂架构设计、性能优化、疑难问题排查，需要深入理解。",
}


# ============================================================
# 4. LLM API 调用
# ============================================================

def create_llm_client(cfg: Dict):
    """
    创建 OpenAI 兼容的 LLM 客户端。

    优先从环境变量读取配置，回退到 config/config.yaml。
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai 包未安装。请执行: pip install openai")
        sys.exit(1)

    base_url = os.environ.get("LLM_API_BASE") or get(cfg, "data.llm_api.base_url", "https://api.openai.com/v1")
    api_key = os.environ.get("LLM_API_KEY") or get(cfg, "data.llm_api.api_key", "")

    if not api_key:
        logger.error(
            "未配置 LLM API Key。请设置环境变量 LLM_API_KEY 或在 config/config.yaml 中设置 data.llm_api.api_key"
        )
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)
    model = os.environ.get("LLM_MODEL") or get(cfg, "data.llm_api.model", "gpt-4o")
    logger.info("LLM API 配置: base_url=%s, model=%s", base_url, model)
    return client, model


def _build_prompt(
    category_def: dict,
    seed_example: Dict,
    subcategory: str,
    difficulty: str,
    batch_size: int = 1,
) -> str:
    """
    构建 meta-prompt。

    参数:
        category_def: 从 YAML 加载的类型定义 dict，包含 meta_prompt, extra_instructions 等
        seed_example: 种子样本 dict
        subcategory: 当前子话题名称
        difficulty: 难度级别
        batch_size: 批量生成数量
    """
    template = category_def["meta_prompt"]

    # 准备种子示例（去掉 metadata，只保留 messages）
    seed_messages_only = {"messages": seed_example["messages"]}
    seed_json = json.dumps(seed_messages_only, ensure_ascii=False, indent=2)

    # 截断过长的种子示例
    if len(seed_json) > 6000:
        seed_json = seed_json[:6000] + "\n... (示例已截断)"

    difficulty_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["intermediate"])

    # 从类型定义中获取 extra_instruction
    extra_instructions = category_def.get("extra_instructions", {})
    extra_instruction = extra_instructions.get(subcategory, "")

    kwargs = {
        "seed_example": seed_json,
        "difficulty_instruction": difficulty_instruction,
        "subcategory": subcategory,
        "tool_definitions": TOOL_DEFINITIONS_TEXT,
        "extra_instruction": extra_instruction,
        "batch_size": batch_size,
    }

    return template.format(**kwargs)


def generate_via_llm(
    client,
    model: str,
    category: str,
    category_def: dict,
    seeds: List[Dict],
    count: int = 10,
    batch_size: int = 5,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[Dict]:
    """
    调用 LLM API 批量生成训练数据。

    参数：
        client: OpenAI 客户端
        model: 模型名称
        category: 类别名称
        category_def: 从 YAML 加载的类型定义 dict
        seeds: 该类别的种子样本列表
        count: 目标生成数量
        batch_size: 每次 API 调用生成的样本数（默认 5）
        max_retries: 每个请求的最大重试次数
        retry_delay: 重试间隔（秒）

    返回：
        生成的样本列表（已通过格式验证）
    """
    if not seeds:
        logger.warning("类别 %s 没有种子样本，跳过生成", category)
        return []

    subcategories = category_def.get("subcategories", ["general"])
    generated: List[Dict] = []
    failed_count = 0
    api_call_count = 0

    # 计算需要的 API 调用次数
    total_calls = math.ceil(count / batch_size)
    max_tokens = min(batch_size * 4096, 16384)

    logger.info(
        "开始生成类别 '%s' 的 %d 条数据 (batch_size=%d, 预计 API 调用 %d 次, max_tokens=%d)...",
        category, count, batch_size, total_calls, max_tokens,
    )

    for i in range(total_calls):
        # 已收集够了，提前退出
        if len(generated) >= count:
            break

        # 轮询子话题和难度
        subcategory = subcategories[i % len(subcategories)]
        difficulty = DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)]

        # 随机选择种子示例
        seed_example = random.choice(seeds)

        prompt = _build_prompt(
            category_def, seed_example, subcategory, difficulty,
            batch_size=batch_size,
        )

        batch_success = 0
        batch_fail = 0
        for attempt in range(max_retries):
            try:
                api_call_count += 1
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一位训练数据生成专家。请严格按照要求输出合法的 JSON 数组，"
                                "数组中每个元素都是一个独立的训练样本。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                    max_tokens=max_tokens,
                )

                raw_text = response.choices[0].message.content.strip()
                parsed_samples = _parse_llm_response(raw_text)

                if parsed_samples:
                    # 对每条样本独立验证
                    for sample in parsed_samples:
                        valid, errors = validate_sample(sample)
                        if valid:
                            sample["metadata"] = {
                                "category": category,
                                "subcategory": subcategory,
                                "difficulty": difficulty,
                                "source": "llm_generated",
                            }
                            generated.append(sample)
                            batch_success += 1
                        else:
                            batch_fail += 1
                            logger.debug(
                                "批次 %d 中一条样本验证失败: %s",
                                i + 1, "; ".join(errors),
                            )
                    # 只要有成功的就跳出重试
                    if batch_success > 0:
                        break
                else:
                    logger.debug(
                        "批次 %d JSON 解析失败 (尝试 %d/%d)",
                        i + 1, attempt + 1, max_retries,
                    )

            except Exception as e:
                logger.warning(
                    "API 调用失败 (尝试 %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        if batch_success == 0:
            failed_count += batch_size  # 整批失败

        logger.info(
            "  批次 %d/%d: 解析 %d 条, 验证通过 %d, 失败 %d (累计成功 %d/%d)",
            i + 1, total_calls, batch_success + batch_fail,
            batch_success, batch_fail, len(generated), count,
        )

    # 如果生成的超过了目标数量，截断
    if len(generated) > count:
        generated = generated[:count]

    total_attempted = len(generated) + failed_count
    success_rate = len(generated) / total_attempted * 100 if total_attempted > 0 else 0

    logger.info(
        "类别 '%s' 生成完成: 成功 %d/%d, API 调用 %d 次, 成功率 %.1f%%",
        category, len(generated), count, api_call_count, success_rate,
    )
    return generated


def _parse_llm_response(text: str) -> List[Dict]:
    """
    解析 LLM 响应文本为样本列表。

    支持三种情况：
    1. JSON 数组 [{...}, {...}, ...] — 批量生成的标准返回格式
    2. 单个 JSON 对象 {...} — 向后兼容，包装为 [dict]
    3. 截断的 JSON 数组 — 降级方案，逐个提取 {"messages": [...]} 对象

    处理常见的 LLM 输出问题：
    - markdown 代码块包裹
    - 前后有多余文字
    - 不完整的 JSON

    返回:
        解析出的样本列表（每个元素是含 messages 字段的 dict）。
        解析失败时返回空列表 []。
    """
    # 去掉 markdown 代码块标记
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # === 策略 1：尝试直接解析完整 JSON ===
    try:
        data = json.loads(text)
        # 情况 1a：JSON 数组
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, dict) and "messages" in item:
                    results.append(item)
            if results:
                return results
        # 情况 1b：单个 JSON 对象（向后兼容）
        if isinstance(data, dict) and "messages" in data:
            return [data]
    except json.JSONDecodeError:
        pass

    # === 策略 2：尝试提取 JSON 数组 [...] ===
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        try:
            data = json.loads(text[first_bracket : last_bracket + 1])
            if isinstance(data, list):
                results = []
                for item in data:
                    if isinstance(item, dict) and "messages" in item:
                        results.append(item)
                if results:
                    return results
        except json.JSONDecodeError:
            pass

    # === 策略 3：尝试提取单个 JSON 对象 {...} ===
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(text[first_brace : last_brace + 1])
            if isinstance(data, dict) and "messages" in data:
                return [data]
        except json.JSONDecodeError:
            pass

    # === 策略 4（降级）：正则逐个提取 {"messages": [...]} 对象 ===
    # 当 JSON 数组被截断时，尝试从响应中逐个匹配完整的 {"messages": ...} 对象
    results = _extract_individual_samples(text)
    if results:
        return results

    return []


def _extract_individual_samples(text: str) -> List[Dict]:
    """
    降级提取方案：从可能截断的文本中逐个提取 {"messages": [...]} JSON 对象。

    使用括号匹配而非正则来处理嵌套 JSON。
    """
    results = []
    # 查找所有 {"messages" 的起始位置
    pattern = re.compile(r'\{\s*"messages"\s*:')
    for match in pattern.finditer(text):
        start = match.start()
        # 尝试从这个位置开始匹配完整的 JSON 对象（括号配对）
        depth = 0
        end = start
        in_string = False
        escape_next = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if depth == 0 and end > start:
            try:
                obj = json.loads(text[start:end])
                if isinstance(obj, dict) and "messages" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                pass
    return results


# ============================================================
# 5. 输出写入
# ============================================================

def write_samples(
    samples: List[Dict],
    output_path: str,
    strip_metadata: bool = False,
) -> int:
    """
    将样本写入 JSONL 文件。

    参数：
        samples: 样本列表
        output_path: 输出文件路径
        strip_metadata: 是否去掉 metadata 字段（写训练数据时应去掉）

    返回：
        写入的样本数
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            if strip_metadata:
                # 只保留 messages 字段
                out = {"messages": sample["messages"]}
            else:
                out = sample
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    logger.info("已写入 %d 条样本到 %s", count, output_path)
    return count


# ============================================================
# 6. 旧版兼容：从种子生成训练数据（无 LLM）
# ============================================================

def generate_from_seeds_only(
    seeds: Dict[str, List[Dict]],
    target_size: int = 800,
    ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    仅使用种子样本生成训练数据（不调用 LLM API）。

    通过重复和打散种子样本填充到目标数量。
    这是旧版 generate_data.py 行为的兼容实现。

    参数：
        seeds: {category: [samples]} 字典
        target_size: 目标样本总数
        ratios: 各类别的比例，如 {"java_coding": 0.5, ...}
        seed: 随机种子
    """
    random.seed(seed)

    all_samples = []
    for category, samples in seeds.items():
        for sample in samples:
            # 去掉 metadata 保持兼容
            all_samples.append({"messages": sample["messages"]})

    base_count = len(all_samples)
    logger.info("种子样本总数: %d", base_count)

    if base_count == 0:
        logger.warning("没有种子样本，无法生成训练数据")
        return []

    # 扩充到目标数量
    if base_count < target_size:
        multiplier = (target_size // base_count) + 1
        expanded = []
        for _ in range(multiplier):
            batch = all_samples.copy()
            random.shuffle(batch)
            expanded.extend(batch)
        all_samples = expanded[:target_size]

    random.shuffle(all_samples)
    logger.info("生成 %d 条训练样本（种子扩充模式）", len(all_samples))
    return all_samples


# ============================================================
# 7. CLI 主入口
# ============================================================

def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    """解析 CLI 特殊参数（--count, --category, --categories, --batch-size, --merge, --review, --report 等）。"""
    result: Dict[str, Any] = {
        "merge": False,
        "review": False,
        "report": False,
        "count": None,
        "category": None,
        "categories": None,  # 逗号分隔的多类型列表
        "batch_size": None,
    }
    for i, arg in enumerate(argv):
        if arg == "--merge":
            result["merge"] = True
        elif arg == "--review":
            result["review"] = True
        elif arg == "--report":
            result["report"] = True
        elif arg == "--count" and i + 1 < len(argv):
            try:
                result["count"] = int(argv[i + 1])
            except (ValueError, IndexError):
                pass
        elif arg == "--category" and i + 1 < len(argv):
            result["category"] = argv[i + 1]
        elif arg == "--categories" and i + 1 < len(argv):
            # 逗号分隔的多类型列表
            result["categories"] = [c.strip() for c in argv[i + 1].split(",") if c.strip()]
        elif arg == "--batch-size" and i + 1 < len(argv):
            try:
                result["batch_size"] = int(argv[i + 1])
            except (ValueError, IndexError):
                pass
    return result


def main():
    """主入口函数。"""
    # 加载配置（CLI 参数可覆盖 config/config.yaml）
    cfg = load_config(cli_args=sys.argv[1:])

    # 解析 CLI 特殊参数
    cli = _parse_cli_args(sys.argv[1:])

    # 获取配置参数
    count = cli["count"] or get(cfg, "data.sample_count", 2000)
    output_path = Path(SCRIPT_DIR) / get(cfg, "data.training_data_path", "training_data.jsonl")
    seeds_dir = SCRIPT_DIR / get(cfg, "data.seeds_dir", "data/seeds")
    generated_dir = SCRIPT_DIR / get(cfg, "data.generated_dir", "data/generated")
    rejected_path = SCRIPT_DIR / get(cfg, "data.rejected_path", "rejected_samples.jsonl")
    report_path = SCRIPT_DIR / get(cfg, "data.report_path", "data_report.json")

    # 动态加载类型定义
    categories_dir = get(cfg, "data.categories_dir", "data/categories")
    category_defs = load_categories(categories_dir)
    valid_categories = list(category_defs.keys())

    # 解析类型列表: --categories 优先，其次 --category，最后从 config 推导
    if cli["categories"]:
        requested_categories = cli["categories"]
    elif cli["category"]:
        requested_categories = [cli["category"]]
    else:
        requested_categories = None  # 后续从 config ratios 推导

    # 验证指定的类别
    if requested_categories:
        for cat in requested_categories:
            if cat not in valid_categories:
                logger.error(
                    "无效的类别: '%s'。可用类型: %s",
                    cat, ", ".join(valid_categories),
                )
                sys.exit(1)
            # 检查种子数据（警告但不阻塞）
            seed_file = seeds_dir / f"{cat}.jsonl"
            if not seed_file.exists():
                logger.warning(
                    "类别 '%s' 没有种子数据文件 (%s)，LLM 可以零种子生成但质量可能降低",
                    cat, seed_file,
                )

    # batch_size: CLI > config > 默认 5
    batch_size = cli["batch_size"] or get(cfg, "data.batch_size", 5)

    logger.info("=" * 60)
    logger.info("QLoRA 训练数据生成器")
    logger.info("=" * 60)

    # --report 模式：生成数据质量报告
    if cli["report"]:
        logger.info("模式: 数据质量报告 (--report)")
        generate_data_report(
            seeds_dir=seeds_dir,
            generated_dir=generated_dir,
            output_path=report_path,
        )
        return

    # --review 模式：交互式人工审查
    if cli["review"]:
        logger.info("模式: 交互式人工审查 (--review)")
        interactive_review(
            seeds_dir=seeds_dir,
            generated_dir=generated_dir,
            rejected_path=rejected_path,
        )
        return

    # --merge 模式：合并所有数据（排除审查失败和被拒绝的样本）
    if cli["merge"]:
        logger.info("模式: 数据合并 (--merge)")
        ratios = get(cfg, "data.category_ratios", {
            "java_coding": 0.50,
            "tool_calling": 0.30,
            "frontend_dev": 0.20,
        })
        similarity_threshold = get(cfg, "data.dedup_similarity_threshold", 0.85)
        merge_seed = get(cfg, "data.split_seed", 42)

        # 同时生成报告
        generate_data_report(
            seeds_dir=seeds_dir,
            generated_dir=generated_dir,
            output_path=report_path,
        )

        # 使用审查后的合并流程
        merge_reviewed_data(
            seeds_dir=seeds_dir,
            generated_dir=generated_dir,
            rejected_path=rejected_path,
            output_path=output_path,
            category_ratios=ratios,
            target_size=None,
            similarity_threshold=similarity_threshold,
            seed=merge_seed,
        )
        return

    # 生成模式
    # 加载种子数据
    seeds = load_seeds(str(seeds_dir.relative_to(SCRIPT_DIR)))

    if not seeds:
        logger.error("未找到种子样本。请确认 %s 目录存在且包含 .jsonl 文件", seeds_dir)
        sys.exit(1)

    total_seeds = sum(len(v) for v in seeds.values())
    logger.info("种子样本: %d 条 (%s)",
                total_seeds,
                ", ".join(f"{k}={len(v)}" for k, v in seeds.items()))

    # 判断是否有 LLM API 配置
    api_key = os.environ.get("LLM_API_KEY") or get(cfg, "data.llm_api.api_key", "")
    use_llm = bool(api_key)

    if not use_llm:
        # 回退到种子扩充模式
        logger.warning("未配置 LLM API Key，使用种子样本扩充模式（重复+打散）")
        logger.warning("设置环境变量 LLM_API_KEY 或在 config/config.yaml 中配置 data.llm_api.api_key 以启用 LLM 生成")

        ratios = get(cfg, "data.category_ratios", None)
        training_seed = get(cfg, "training.seed", 42)
        samples = generate_from_seeds_only(seeds, target_size=count, ratios=ratios, seed=training_seed)

        if samples:
            write_samples(samples, str(output_path), strip_metadata=True)

            # 验证
            passed, failed, errors = validate_batch(samples)
            logger.info("验证结果: 通过 %d, 失败 %d", passed, failed)
            if errors:
                for e in errors[:10]:
                    logger.warning("  %s", e)
        return

    # LLM 生成模式
    logger.info("使用 LLM API 生成训练数据 (batch_size=%d)", batch_size)
    client, model = create_llm_client(cfg)

    # 按类别分配数量
    ratios = get(cfg, "data.category_ratios", {
        "java_coding": 0.50,
        "tool_calling": 0.30,
        "frontend_dev": 0.20,
    })

    if requested_categories and len(requested_categories) == 1:
        # 只生成指定的单个类别
        categories_to_generate = {requested_categories[0]: count}
    elif requested_categories:
        # 指定了多个类别，按 ratios 比例分配；不在 ratios 中的类别平均分配剩余
        categories_to_generate = {}
        # 计算有 ratio 的类别和没有 ratio 的类别
        with_ratio = {c: ratios[c] for c in requested_categories if c in ratios}
        without_ratio = [c for c in requested_categories if c not in ratios]

        # 分配有 ratio 的类别
        total_ratio = sum(with_ratio.values())
        if total_ratio > 0:
            for cat, ratio in with_ratio.items():
                # 按比例在指定的类别中分配
                normalized_ratio = ratio / total_ratio
                if without_ratio:
                    # 留出一部分给没有 ratio 的类别
                    normalized_ratio *= (1 - len(without_ratio) / len(requested_categories))
                categories_to_generate[cat] = max(1, int(count * normalized_ratio))

        # 平均分配给没有 ratio 的类别
        if without_ratio:
            assigned = sum(categories_to_generate.values())
            remaining = max(0, count - assigned)
            per_cat = max(1, remaining // len(without_ratio))
            for cat in without_ratio:
                categories_to_generate[cat] = per_cat
    else:
        # 未指定类别，从 ratios 的 keys 推导
        categories_to_generate = {}
        for cat, ratio in ratios.items():
            cat_count = max(1, int(count * ratio))
            categories_to_generate[cat] = cat_count

    # 断点续传：检查已有样本数，计算还需生成多少
    for cat in list(categories_to_generate.keys()):
        existing = count_existing_samples(generated_dir, category=cat)
        target = categories_to_generate[cat]
        if existing >= target:
            logger.info(
                "类别 '%s' 已有 %d 条样本（目标 %d），跳过生成",
                cat, existing, target,
            )
            del categories_to_generate[cat]
        elif existing > 0:
            remaining = target - existing
            logger.info(
                "类别 '%s' 已有 %d 条样本（目标 %d），继续生成 %d 条",
                cat, existing, target, remaining,
            )
            categories_to_generate[cat] = remaining

    if not categories_to_generate:
        logger.info("所有类别均已达到目标数量，无需生成")
        logger.info("提示：使用 --merge 合并数据生成 training_data.jsonl")
        return

    for cat, cat_count in categories_to_generate.items():
        cat_seeds = seeds.get(cat, [])
        if not cat_seeds:
            logger.warning("类别 '%s' 没有种子样本，跳过", cat)
            continue

        # 获取类型定义
        cat_def = category_defs.get(cat)
        if cat_def is None:
            logger.error("类别 '%s' 没有对应的 YAML 类型定义，跳过", cat)
            continue

        generated = generate_via_llm(
            client=client,
            model=model,
            category=cat,
            category_def=cat_def,
            seeds=cat_seeds,
            count=cat_count,
            batch_size=batch_size,
        )

        if generated:
            # 写入到 data/generated/ 目录，文件名格式：{category}_{batch_number:03d}.jsonl
            generated_dir.mkdir(parents=True, exist_ok=True)
            batch_num = get_next_batch_number(generated_dir, cat)
            batch_filename = f"{cat}_{batch_num:03d}.jsonl"
            batch_path = generated_dir / batch_filename
            write_samples(generated, str(batch_path), strip_metadata=False)

            # 验证统计
            passed, failed, errors = validate_batch(generated)
            logger.info(
                "类别 '%s' 批次 %s: %d 条 (验证通过 %d, 失败 %d)",
                cat, batch_filename, len(generated), passed, failed,
            )
            if errors:
                for e in errors[:5]:
                    logger.warning("  %s", e)

    # 生成完成后提示合并
    total_generated = count_existing_samples(generated_dir)
    logger.info("=" * 60)
    logger.info("生成完成。data/generated/ 共有 %d 条样本", total_generated)
    logger.info("使用 'python generate_data.py --merge' 合并数据生成 training_data.jsonl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
