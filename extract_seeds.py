"""
临时脚本：从 generate_data.py 中提取所有手写样本到 data/seeds/ 目录下的 JSONL 文件。
运行后可删除此脚本。

用法: python extract_seeds.py
"""

import json
import sys
import os

# 确保可以导入 generate_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import (
    generate_springboot_basics,
    generate_springboot_web,
    generate_spring_cloud,
    generate_mybatis,
    generate_flyway,
    generate_webflux,
    generate_exception_diagnosis,
    generate_code_review,
    generate_simple_tool_calls,
    generate_multi_step_planning,
    generate_tool_constraint_samples,
    generate_error_recovery_samples,
    generate_refusal_samples,
    generate_param_reflection,
)

SEEDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seeds")


def add_metadata(sample: dict, category: str, subcategory: str, difficulty: str) -> dict:
    """给样本添加元数据标签"""
    return {
        "messages": sample["messages"],
        "metadata": {
            "category": category,
            "subcategory": subcategory,
            "difficulty": difficulty,
        }
    }


def write_jsonl(filepath: str, samples: list):
    """写入 JSONL 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  写入 {len(samples)} 条样本到 {filepath}")


def extract_java_coding():
    """提取 Java 编程类种子样本"""
    samples = []

    # Spring Boot 基础 (4 samples)
    for s in generate_springboot_basics():
        samples.append(add_metadata(s, "java_coding", "springboot_basics", "intermediate"))

    # Spring Boot Web开发 (3 samples)
    for s in generate_springboot_web():
        samples.append(add_metadata(s, "java_coding", "springboot_web", "intermediate"))

    # Spring Cloud 微服务 (3 samples)
    for s in generate_spring_cloud():
        samples.append(add_metadata(s, "java_coding", "spring_cloud", "advanced"))

    # MyBatis (3 samples)
    for s in generate_mybatis():
        samples.append(add_metadata(s, "java_coding", "mybatis", "intermediate"))

    # Flyway (1 sample)
    for s in generate_flyway():
        samples.append(add_metadata(s, "java_coding", "flyway", "intermediate"))

    # WebFlux (2 samples)
    for s in generate_webflux():
        samples.append(add_metadata(s, "java_coding", "webflux", "advanced"))

    # 异常排查 (4 samples)
    for s in generate_exception_diagnosis():
        samples.append(add_metadata(s, "java_coding", "exception_diagnosis", "advanced"))

    # 代码审查/重构 (1 sample)
    for s in generate_code_review():
        samples.append(add_metadata(s, "java_coding", "code_review", "intermediate"))

    return samples


def extract_tool_calling():
    """提取工具调用类种子样本"""
    samples = []

    # 简单工具调用 (3 samples)
    for s in generate_simple_tool_calls():
        samples.append(add_metadata(s, "tool_calling", "simple_tool_call", "basic"))

    # 多步骤任务规划 (2 samples)
    for s in generate_multi_step_planning():
        samples.append(add_metadata(s, "tool_calling", "multi_step_planning", "advanced"))

    # 工具约束遵守 (2 samples)
    for s in generate_tool_constraint_samples():
        samples.append(add_metadata(s, "tool_calling", "tool_constraint", "intermediate"))

    # 异常处理链路 (1 sample)
    for s in generate_error_recovery_samples():
        samples.append(add_metadata(s, "tool_calling", "error_recovery", "advanced"))

    # 拒绝/降级场景 (2 samples)
    for s in generate_refusal_samples():
        # 第一个是纯知识问答（无工具），第二个是工具调用
        samples.append(add_metadata(s, "tool_calling", "refusal_or_fallback", "basic"))

    return samples


def extract_param_reflection():
    """提取参数反思类种子样本"""
    samples = []
    all_reflection = generate_param_reflection()

    # 根据注释中的分类标记 subcategory
    # 样本顺序（从generate_param_reflection源码分析）:
    # 0-2: 参数类型校验 (type_check)
    # 3-4: 必填参数检查 (required_check)
    # 5-7: 参数值合理性 (value_reasonableness)
    # 8-9: 工具选择推理 (tool_selection)
    # 10-11: 错误参数自纠正 (error_correction)
    # 12-14: 执行后果预估 (execution_impact)

    subcategory_map = [
        ("type_check", "basic"),           # 0: 看项目配置文件
        ("type_check", "basic"),           # 1: 搜索@RestController
        ("type_check", "basic"),           # 2: 列出根目录文件
        ("required_check", "intermediate"),  # 3: 升级pom版本
        ("required_check", "intermediate"),  # 4: 创建Flyway迁移脚本
        ("value_reasonableness", "intermediate"),  # 5: 改@Autowired为构造器注入
        ("value_reasonableness", "intermediate"),  # 6: 搜索"user name"(含空格)
        ("value_reasonableness", "intermediate"),  # 7: 运行scripts/deploy.sh
        ("tool_selection", "intermediate"),   # 8: 搜索logger变量
        ("tool_selection", "intermediate"),   # 9: 读取含空格路径文件
        ("error_correction", "advanced"),   # 10: 看OrderService.create方法
        ("error_correction", "advanced"),   # 11: 查看项目第三方依赖
        ("execution_impact", "intermediate"),  # 12: 替换System.out.println为log.info
        ("execution_impact", "intermediate"),  # 13: 搜索硬编码密码
        ("execution_impact", "intermediate"),  # 14: Maven编译
        ("execution_impact", "intermediate"),  # 15: 后台启动Spring Boot
        ("execution_impact", "intermediate"),  # 16: 检查未使用import
        ("execution_impact", "intermediate"),  # 17: 长时间数据迁移脚本
        ("execution_impact", "intermediate"),  # 18: 数据库备份
        ("execution_impact", "intermediate"),  # 19: 运行mvn test
        ("execution_impact", "intermediate"),  # 20: ab压力测试
    ]

    for i, s in enumerate(all_reflection):
        if i < len(subcategory_map):
            subcat, diff = subcategory_map[i]
        else:
            subcat, diff = "general", "intermediate"
        samples.append(add_metadata(s, "param_reflection", subcat, diff))

    return samples


def main():
    os.makedirs(SEEDS_DIR, exist_ok=True)

    print("正在从 generate_data.py 提取种子样本...")
    print()

    # 提取各类别
    java_samples = extract_java_coding()
    tool_samples = extract_tool_calling()
    reflection_samples = extract_param_reflection()

    print(f"Java 编程样本: {len(java_samples)}")
    write_jsonl(os.path.join(SEEDS_DIR, "java_coding.jsonl"), java_samples)

    print(f"工具调用样本: {len(tool_samples)}")
    write_jsonl(os.path.join(SEEDS_DIR, "tool_calling.jsonl"), tool_samples)

    print(f"参数反思样本: {len(reflection_samples)}")
    write_jsonl(os.path.join(SEEDS_DIR, "param_reflection.jsonl"), reflection_samples)

    total = len(java_samples) + len(tool_samples) + len(reflection_samples)
    print(f"\n总计: {total} 条种子样本已提取")

    # 验证所有文件
    print("\n正在验证 JSONL 文件格式...")
    for filename in ["java_coding.jsonl", "tool_calling.jsonl", "param_reflection.jsonl"]:
        filepath = os.path.join(SEEDS_DIR, filename)
        errors = 0
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                count += 1
                try:
                    data = json.loads(line)
                    # 检查必需字段
                    assert "messages" in data, f"缺少 messages 字段"
                    assert "metadata" in data, f"缺少 metadata 字段"
                    assert len(data["messages"]) >= 2, f"消息数量不足"
                    assert "category" in data["metadata"], f"缺少 category"
                    assert "subcategory" in data["metadata"], f"缺少 subcategory"
                    assert "difficulty" in data["metadata"], f"缺少 difficulty"
                    # 检查 messages 格式
                    for msg in data["messages"]:
                        assert "role" in msg, f"消息缺少 role"
                        assert "content" in msg, f"消息缺少 content"
                        assert msg["role"] in ("system", "user", "assistant"), f"无效的 role: {msg['role']}"
                except (json.JSONDecodeError, AssertionError) as e:
                    errors += 1
                    print(f"  错误 {filename}:{line_num}: {e}")

        status = "✓ 通过" if errors == 0 else f"✗ {errors} 个错误"
        print(f"  {filename}: {count} 条, {status}")


if __name__ == "__main__":
    main()
