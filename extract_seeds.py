"""
种子数据管理脚本。

功能：
- 验证 data/seeds/ 目录下所有 JSONL 种子文件的格式
- 如果旧版 generate_data.py 中存在内联样本生成函数，可提取到 JSONL 文件
  （当前版本已迁移为插件式类型定义，内联函数已移除）

用法: python extract_seeds.py
"""

import json
import sys
import os

# 确保可以导入 generate_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEEDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seeds")


def validate_jsonl(filepath: str) -> tuple:
    """验证 JSONL 文件格式，返回 (count, errors)。"""
    if not os.path.exists(filepath):
        return 0, 1

    errors = 0
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            count += 1
            try:
                data = json.loads(line)
                assert "messages" in data, "缺少 messages 字段"
                assert "metadata" in data, "缺少 metadata 字段"
                assert len(data["messages"]) >= 2, "消息数量不足"
                assert "category" in data["metadata"], "缺少 category"
                assert "subcategory" in data["metadata"], "缺少 subcategory"
                assert "difficulty" in data["metadata"], "缺少 difficulty"
                for msg in data["messages"]:
                    assert "role" in msg, "消息缺少 role"
                    assert "content" in msg, "消息缺少 content"
                    assert msg["role"] in ("system", "user", "assistant"), f"无效的 role: {msg['role']}"
            except (json.JSONDecodeError, AssertionError) as e:
                errors += 1
                print(f"  错误 {os.path.basename(filepath)}:{line_num}: {e}")
    return count, errors


def main():
    os.makedirs(SEEDS_DIR, exist_ok=True)

    # 期望的种子文件列表
    expected_files = ["java_coding.jsonl", "tool_calling.jsonl", "frontend_dev.jsonl"]

    print("正在验证种子数据文件...")
    print()

    total_count = 0
    total_errors = 0

    for filename in expected_files:
        filepath = os.path.join(SEEDS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠ {filename} 不存在")
            total_errors += 1
            continue

        count, errors = validate_jsonl(filepath)
        total_count += count
        total_errors += errors

        status = "✓ 通过" if errors == 0 else f"✗ {errors} 个错误"
        print(f"  {filename}: {count} 条, {status}")

    print(f"\n总计: {total_count} 条种子样本, {total_errors} 个错误")

    if total_errors > 0:
        print("\n存在验证错误，请检查上述文件。")
        sys.exit(1)
    else:
        print("\n所有种子文件验证通过。")


if __name__ == "__main__":
    main()
