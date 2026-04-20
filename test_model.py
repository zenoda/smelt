"""
测试脚本：验证模型微调效果
测试维度：Java编程能力 + 工具调用能力 + 任务规划能力 + 参数反思能力

支持功能：
- --model <path>：指定模型路径（合并模型目录 或 LoRA 适配器目录）
- --compare：对比模式，依次加载原始模型和微调模型，输出对比报告
- --output <path>：指定测试结果输出文件路径
- 自动检测 LoRA 适配器：如果指定路径包含 adapter_config.json，
  则自动加载基础模型 + LoRA 适配器（无需预先合并）
- 从 config/config.yaml 读取默认路径配置
"""

import argparse
import gc
import os
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import sys

from config_utils import load_config, get


TOOL_DEFINITIONS_STR = """[
  {"type": "function", "function": {"name": "read_file", "description": "读取指定路径的文件内容", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
  {"type": "function", "function": {"name": "edit_file", "description": "通过精确字符串匹配替换来编辑文件", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
  {"type": "function", "function": {"name": "run_command", "description": "执行shell命令", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "workdir": {"type": "string"}}, "required": ["command"]}}},
  {"type": "function", "function": {"name": "search_files", "description": "搜索文件内容", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}}
]"""


def is_lora_adapter(model_path: str) -> bool:
    """检查给定路径是否为 LoRA 适配器目录（而非完整模型）"""
    adapter_config = Path(model_path) / "adapter_config.json"
    return adapter_config.exists()


def get_base_model_from_adapter(adapter_path: str) -> str:
    """从 LoRA 适配器的 adapter_config.json 中读取基础模型路径"""
    config_file = Path(adapter_path) / "adapter_config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    return adapter_cfg.get("base_model_name_or_path", "")


def load_model(model_path: str, base_model_path: str = None):
    """
    加载模型和分词器。

    支持两种模式：
    1. 完整模型：model_path 指向合并后的完整模型目录
    2. LoRA 适配器：model_path 指向 LoRA 适配器目录，自动加载基础模型并合并适配器

    参数：
        model_path: 模型路径（完整模型目录 或 LoRA 适配器目录）
        base_model_path: 基础模型路径（仅在 LoRA 模式时使用，默认从 adapter_config.json 读取）

    返回：
        (model, tokenizer)
    """
    if is_lora_adapter(model_path):
        # LoRA 适配器模式：加载基础模型 + 适配器
        if base_model_path is None:
            base_model_path = get_base_model_from_adapter(model_path)
        if not base_model_path:
            print("错误: LoRA 适配器未记录基础模型路径，请通过 config/config.yaml 的 model.path 指定")
            sys.exit(1)

        print(f"检测到 LoRA 适配器: {model_path}")
        print(f"加载基础模型: {base_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        print(f"加载 LoRA 适配器: {model_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("LoRA 适配器已合并到基础模型（内存中合并，不保存到磁盘）")

    else:
        # 完整模型模式：直接加载
        print(f"加载完整模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    model.eval()
    print("模型加载完成!")
    return model, tokenizer


def release_model(model, tokenizer):
    """
    释放模型和分词器占用的 GPU 和 CPU 内存。
    在对比模式下，加载下一个模型前必须调用此函数以避免 OOM。
    """
    print("释放模型内存...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"GPU 显存释放完成，当前占用: {vram_used:.2f} GB")
    else:
        print("GPU 显存释放完成")


def generate(messages, model, tokenizer, max_new_tokens=1024, tools=None):
    """生成文本"""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools

    text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response


# ============================================================
# 测试函数 - 每个返回结构化的 results 列表和维度分数
# ============================================================


def _check_code_blocks(response: str) -> dict:
    """
    检查回复中代码块的完整性。

    返回:
        dict: {
            "has_code_block": bool,       # 是否包含代码块
            "code_blocks_count": int,     # 代码块数量
            "all_closed": bool,           # 所有代码块是否正确闭合
            "has_language_tag": bool,      # 是否包含语言标记（如 ```java）
        }
    """
    # 查找所有 ``` 的位置
    backtick_positions = [m.start() for m in re.finditer(r'```', response)]
    has_code_block = len(backtick_positions) >= 2
    # 代码块成对闭合
    all_closed = len(backtick_positions) % 2 == 0 and len(backtick_positions) >= 2
    code_blocks_count = len(backtick_positions) // 2

    # 检查语言标记
    has_language_tag = bool(re.search(r'```\w+', response))

    return {
        "has_code_block": has_code_block,
        "code_blocks_count": code_blocks_count,
        "all_closed": all_closed,
        "has_language_tag": has_language_tag,
    }


def _check_java_annotations_apis(response: str, expected_annotations: list, expected_apis: list) -> dict:
    """
    检查回复中关键注解和 API 的使用正确性。

    参数:
        response: 模型回复文本
        expected_annotations: 期望出现的注解列表（如 ["@RestControllerAdvice", "@ExceptionHandler"]）
        expected_apis: 期望出现的 API/类名列表（如 ["MethodArgumentNotValidException"]）

    返回:
        dict: {
            "annotation_hits": list,    # 命中的注解
            "annotation_total": int,    # 期望的注解总数
            "annotation_rate": float,   # 注解命中率 0-1
            "api_hits": list,           # 命中的 API
            "api_total": int,           # 期望的 API 总数
            "api_rate": float,          # API 命中率 0-1
        }
    """
    resp_lower = response.lower()
    ann_hits = [a for a in expected_annotations if a.lower() in resp_lower]
    api_hits = [a for a in expected_apis if a.lower() in resp_lower]

    return {
        "annotation_hits": ann_hits,
        "annotation_total": len(expected_annotations),
        "annotation_rate": len(ann_hits) / max(len(expected_annotations), 1),
        "api_hits": api_hits,
        "api_total": len(expected_apis),
        "api_rate": len(api_hits) / max(len(expected_apis), 1),
    }


def test_java_coding(model, tokenizer, verbose=True):
    """
    测试Java编程能力，返回 (results, score_0_to_100)。

    评分维度（每个用例）：
    - 关键词命中率 (40%): 原有的核心关键词检查
    - 代码块完整性 (30%): 是否包含完整闭合的代码块、语言标记
    - 注解/API 正确性 (30%): 关键 Spring/MyBatis 注解和 API 是否正确使用
    """
    test_cases = [
        {
            "prompt": "写一个Spring Boot的全局异常处理器，要求处理参数校验异常和自定义业务异常。",
            "check_keywords": ["@RestControllerAdvice", "@ExceptionHandler", "MethodArgumentNotValidException"],
            "expected_annotations": ["@RestControllerAdvice", "@ExceptionHandler"],
            "expected_apis": ["MethodArgumentNotValidException", "ResponseEntity", "HttpStatus"],
            "expect_code": True,
            "description": "全局异常处理",
        },
        {
            "prompt": "MyBatis中如何用动态SQL实现条件查询？给出使用<where>和<if>的示例。",
            "check_keywords": ["<where>", "<if", "test="],
            "expected_annotations": ["@Mapper", "@Select"],
            "expected_apis": ["<where>", "<if"],
            "expect_code": True,
            "description": "MyBatis动态SQL",
        },
        {
            "prompt": "Spring Boot项目如何集成Flyway？说明命名规则和关键配置。",
            "check_keywords": ["flyway", "V1__", "migration"],
            "expected_annotations": [],
            "expected_apis": ["flyway", "V1__", "spring.flyway"],
            "expect_code": True,
            "description": "Flyway集成",
        },
        {
            "prompt": "用WebFlux写一个响应式的REST接口，返回Mono和Flux。",
            "check_keywords": ["Mono", "Flux"],
            "expected_annotations": ["@RestController", "@GetMapping"],
            "expected_apis": ["Mono", "Flux", "WebFlux"],
            "expect_code": True,
            "description": "WebFlux响应式接口",
        },
        {
            "prompt": "Spring Cloud OpenFeign怎么声明一个服务调用客户端？包括降级处理。",
            "check_keywords": ["@FeignClient", "fallback"],
            "expected_annotations": ["@FeignClient", "@EnableFeignClients"],
            "expected_apis": ["fallback", "FallbackFactory"],
            "expect_code": True,
            "description": "OpenFeign服务调用",
        },
        {
            "prompt": """应用启动报错：
```
BeanCurrentlyInCreationException: Error creating bean with name 'serviceA':
Is there an unresolvable circular reference?
```
怎么排查和解决？""",
            "check_keywords": ["循环依赖"],
            "expected_annotations": ["@Lazy"],
            "expected_apis": ["循环依赖", "构造器注入"],
            "expect_code": False,
            "description": "循环依赖排查",
        },
        {
            "prompt": "Spring Boot中Filter和Interceptor有什么区别？各适合什么场景？",
            "check_keywords": ["Filter", "Interceptor"],
            "expected_annotations": [],
            "expected_apis": ["Filter", "Interceptor", "HandlerInterceptor"],
            "expect_code": False,
            "description": "Filter vs Interceptor",
        },
        {
            "prompt": """线上报错：
```
HikariPool: Failed to initialize pool: Connection is not available, request timed out after 30000ms
```
可能的原因和解决方案？""",
            "check_keywords": ["连接池", "连接"],
            "expected_annotations": [],
            "expected_apis": ["HikariCP", "maximumPoolSize", "connectionTimeout"],
            "expect_code": False,
            "description": "连接池超时排查",
        },
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("测试1：Java编程能力")
        print("=" * 60)

    results = []
    for i, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\n[{i}/{len(test_cases)}] {tc['description']}")
            print(f"  问题: {tc['prompt'][:80]}...")

        messages = [
            {"role": "system", "content": "你是一位资深Java开发工程师，精通Spring Boot、Spring Cloud、MyBatis、Flyway、WebFlux等框架。"},
            {"role": "user", "content": tc["prompt"]},
        ]

        response = generate(messages, model, tokenizer)
        if verbose:
            print(f"  回复长度: {len(response)} 字符")

        # (1) 关键词命中率 (40%)
        hits = [kw for kw in tc["check_keywords"] if kw.lower() in response.lower()]
        keyword_rate = len(hits) / len(tc["check_keywords"])

        # (2) 代码块完整性 (30%)
        code_info = _check_code_blocks(response)
        if tc["expect_code"]:
            # 期望有代码块时：有代码块 50 分，闭合 30 分，语言标记 20 分
            code_score = 0.0
            if code_info["has_code_block"]:
                code_score += 0.5
            if code_info["all_closed"]:
                code_score += 0.3
            if code_info["has_language_tag"]:
                code_score += 0.2
        else:
            # 不期望代码块时（排查/解释类题），有代码块是加分，没有也不扣分
            code_score = 1.0 if code_info["has_code_block"] and code_info["all_closed"] else 0.7

        # (3) 注解/API 正确性 (30%)
        ann_api_info = _check_java_annotations_apis(
            response,
            tc["expected_annotations"],
            tc["expected_apis"],
        )
        # 注解命中率和 API 命中率加权平均（注解 40%，API 60%）
        if ann_api_info["annotation_total"] > 0 and ann_api_info["api_total"] > 0:
            ann_api_score = ann_api_info["annotation_rate"] * 0.4 + ann_api_info["api_rate"] * 0.6
        elif ann_api_info["api_total"] > 0:
            ann_api_score = ann_api_info["api_rate"]
        elif ann_api_info["annotation_total"] > 0:
            ann_api_score = ann_api_info["annotation_rate"]
        else:
            ann_api_score = 1.0  # 无检查项时满分

        # 综合得分
        case_score = keyword_rate * 0.4 + code_score * 0.3 + ann_api_score * 0.3

        scoring_breakdown = {
            "keyword_rate": round(keyword_rate, 3),
            "keyword_weight": 0.4,
            "code_block_score": round(code_score, 3),
            "code_block_weight": 0.3,
            "code_block_detail": code_info,
            "annotation_api_score": round(ann_api_score, 3),
            "annotation_api_weight": 0.3,
            "annotation_api_detail": ann_api_info,
        }

        if verbose:
            print(f"  关键词命中: {len(hits)}/{len(tc['check_keywords'])} ({keyword_rate:.0%})")
            print(f"  代码块: {'完整' if code_info['all_closed'] else ('有但未闭合' if code_info['has_code_block'] else '无')}"
                  f" ({code_info['code_blocks_count']}个)"
                  f"{' [有语言标记]' if code_info['has_language_tag'] else ''}")
            print(f"  注解/API: 注解 {ann_api_info['annotation_rate']:.0%}"
                  f" ({len(ann_api_info['annotation_hits'])}/{ann_api_info['annotation_total']})"
                  f", API {ann_api_info['api_rate']:.0%}"
                  f" ({len(ann_api_info['api_hits'])}/{ann_api_info['api_total']})")
            print(f"  综合得分: {case_score:.1%}")

        results.append({
            "description": tc["description"],
            "prompt": tc["prompt"],
            "response": response,
            "keywords_total": len(tc["check_keywords"]),
            "keywords_hit": len(hits),
            "score": case_score,
            "scoring_breakdown": scoring_breakdown,
        })

    avg_score = sum(r["score"] for r in results) / len(results)
    dimension_score = round(avg_score * 100, 1)
    if verbose:
        print(f"\nJava编程能力平均得分: {dimension_score}/100")
    return results, dimension_score


def _check_tool_params(response: str, tool_name: str, required_params: list, value_checks: dict = None) -> dict:
    r"""
    检查工具调用中参数的完整性和合理性。

    参数:
        response: 模型回复文本
        tool_name: 期望的工具名
        required_params: 必需参数列表（如 ["path"]）
        value_checks: 参数值合理性检查，格式 {"param_name": regex_pattern}
                      如 {"path": r".*\.xml$", "command": r"mvn\s+"}

    返回:
        dict: {
            "params_found": list,           # 找到的参数名列表
            "params_missing": list,         # 缺失的必需参数
            "param_completeness": float,    # 参数完整率 0-1
            "value_checks_passed": int,     # 通过值合理性检查的数量
            "value_checks_total": int,      # 值合理性检查总数
            "value_reasonableness": float,  # 值合理性得分 0-1
        }
    """
    # 提取所有 <parameter=key>value</parameter> 对
    param_pattern = re.compile(r'<parameter=(\w+)>(.*?)</parameter>', re.DOTALL)
    found_params = {}
    for m in param_pattern.finditer(response):
        found_params[m.group(1)] = m.group(2).strip()

    params_found = list(found_params.keys())
    params_missing = [p for p in required_params if p not in found_params]
    param_completeness = (len(required_params) - len(params_missing)) / max(len(required_params), 1)

    # 值合理性检查
    value_passed = 0
    value_total = 0
    if value_checks:
        for param_name, pattern in value_checks.items():
            value_total += 1
            if param_name in found_params:
                if re.search(pattern, found_params[param_name], re.IGNORECASE):
                    value_passed += 1

    value_reasonableness = value_passed / max(value_total, 1) if value_total > 0 else 1.0

    return {
        "params_found": params_found,
        "params_missing": params_missing,
        "param_completeness": param_completeness,
        "value_checks_passed": value_passed,
        "value_checks_total": value_total,
        "value_reasonableness": value_reasonableness,
    }


def test_tool_calling(model, tokenizer, verbose=True):
    """
    测试工具调用能力，返回 (results, score_0_to_100)。

    评分维度（每个用例）：
    - 工具选择正确性 (35%): 是否调用了正确的工具
    - 格式合规性 (20%): XML 格式是否完整闭合
    - 参数完整性 (25%): 必需参数是否全部提供
    - 参数值合理性 (20%): 参数值是否符合预期格式/内容
    """
    tools = json.loads(TOOL_DEFINITIONS_STR)

    test_cases = [
        {
            "prompt": "帮我看看pom.xml里用了什么Spring Boot版本",
            "expect_tool": "read_file",
            "required_params": ["path"],
            "value_checks": {"path": r"pom\.xml"},
            "description": "简单文件读取",
        },
        {
            "prompt": "搜索项目中所有使用@Transactional的地方",
            "expect_tool": "search_files",
            "required_params": ["pattern"],
            "value_checks": {"pattern": r"@?[Tt]ransactional"},
            "description": "代码搜索",
        },
        {
            "prompt": "执行mvn clean package，看看能不能打包成功",
            "expect_tool": "run_command",
            "required_params": ["command"],
            "value_checks": {"command": r"mvn\s+(clean\s+)?package"},
            "description": "执行命令",
        },
        {
            "prompt": "把application.yml里的server.port从8080改成9090",
            "expect_tool": "read_file",  # 应该先读取再编辑
            "required_params": ["path"],
            "value_checks": {"path": r"application\.ya?ml"},
            "description": "文件编辑（应先读取）",
        },
        {
            "prompt": "Spring Boot中@Autowired和@Resource有什么区别？",
            "expect_tool": None,  # 知识问题，不应调用工具
            "required_params": [],
            "value_checks": {},
            "description": "知识问题（不应调用工具）",
        },
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("测试2：工具调用能力")
        print("=" * 60)

    results = []
    for i, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\n[{i}/{len(test_cases)}] {tc['description']}")
            print(f"  问题: {tc['prompt']}")

        messages = [
            {"role": "system", "content": "你是一位智能编程助手，可以通过调用工具来帮助用户完成软件开发任务。"},
            {"role": "user", "content": tc["prompt"]},
        ]

        response = generate(messages, model, tokenizer, tools=tools)

        # 检查工具调用格式
        has_tool_call = "<tool_call>" in response and "</tool_call>" in response
        tool_format_valid = False
        called_tool = None

        if has_tool_call:
            # 提取工具名
            match = re.search(r'<function=(\w+)>', response)
            if match:
                called_tool = match.group(1)
                # 检查是否有完整的闭合标签
                if '</function>' in response and '</tool_call>' in response:
                    tool_format_valid = True

        # (1) 工具选择正确性 (35%)
        if tc["expect_tool"] is None:
            correct = not has_tool_call
            status = "正确（未调用工具）" if correct else "错误（不应调用工具但调用了）"
            tool_choice_score = 1.0 if correct else 0.0
        else:
            correct = called_tool == tc["expect_tool"]
            if has_tool_call:
                status = f"调用了 {called_tool}" + (" (正确)" if correct else f" (期望 {tc['expect_tool']})")
                tool_choice_score = 1.0 if correct else 0.0
            else:
                status = f"未调用工具（期望 {tc['expect_tool']}）"
                tool_choice_score = 0.0

        # (2) 格式合规性 (20%)
        if tc["expect_tool"] is None:
            format_score = 1.0  # 不需要调用工具时格式默认满分
        elif has_tool_call:
            format_score = 1.0 if tool_format_valid else 0.3  # 有调用但格式不完整
        else:
            format_score = 0.0  # 应该调用但没调用

        # (3) 参数完整性 (25%) 和 (4) 参数值合理性 (20%)
        if tc["expect_tool"] is None or not has_tool_call:
            # 不需要调用工具 或 未调用工具时
            param_info = {
                "params_found": [], "params_missing": [],
                "param_completeness": 1.0 if tc["expect_tool"] is None else 0.0,
                "value_checks_passed": 0, "value_checks_total": 0,
                "value_reasonableness": 1.0 if tc["expect_tool"] is None else 0.0,
            }
        else:
            param_info = _check_tool_params(
                response, called_tool or "",
                tc["required_params"],
                tc.get("value_checks", {}),
            )

        param_completeness_score = param_info["param_completeness"]
        value_reasonableness_score = param_info["value_reasonableness"]

        # 综合得分
        case_score = (
            tool_choice_score * 0.35
            + format_score * 0.20
            + param_completeness_score * 0.25
            + value_reasonableness_score * 0.20
        )

        scoring_breakdown = {
            "tool_choice_score": round(tool_choice_score, 3),
            "tool_choice_weight": 0.35,
            "format_score": round(format_score, 3),
            "format_weight": 0.20,
            "param_completeness_score": round(param_completeness_score, 3),
            "param_completeness_weight": 0.25,
            "value_reasonableness_score": round(value_reasonableness_score, 3),
            "value_reasonableness_weight": 0.20,
            "param_detail": param_info,
        }

        if verbose:
            print(f"  工具调用: {'是' if has_tool_call else '否'}")
            print(f"  格式合规: {'是' if tool_format_valid else '否' if has_tool_call else 'N/A'}")
            print(f"  评估: {status}")
            if has_tool_call and tc["expect_tool"] is not None:
                print(f"  参数完整: {param_info['param_completeness']:.0%}"
                      f" (缺失: {param_info['params_missing'] or '无'})")
                if param_info["value_checks_total"] > 0:
                    print(f"  参数合理: {param_info['value_checks_passed']}/{param_info['value_checks_total']}")
            print(f"  综合得分: {case_score:.1%}")

        results.append({
            "description": tc["description"],
            "prompt": tc["prompt"],
            "response": response[:500],
            "has_tool_call": has_tool_call,
            "tool_format_valid": tool_format_valid,
            "called_tool": called_tool,
            "expected_tool": tc["expect_tool"],
            "correct": correct,
            "score": case_score,
            "scoring_breakdown": scoring_breakdown,
        })

    # 维度得分 = 各用例综合得分的平均值
    dimension_score = round(sum(r["score"] for r in results) / len(results) * 100, 1)

    if verbose:
        correct_count = sum(1 for r in results if r["correct"])
        print(f"\n工具调用正确率: {correct_count}/{len(results)}")
        print(f"工具调用维度得分: {dimension_score}/100")
    return results, dimension_score


def _check_step_numbering(response: str) -> dict:
    """
    检查回复中是否包含清晰的步骤编号。

    返回:
        dict: {
            "has_numbering": bool,       # 是否有编号
            "numbering_style": str,      # 编号风格: "arabic", "chinese", "bullet", "mixed", "none"
            "max_step_number": int,      # 最大步骤编号
            "numbering_consistent": bool, # 编号是否连续
        }
    """
    # 检测不同编号风格
    arabic_steps = re.findall(r'(?:^|\n)\s*(\d+)\s*[.、）)]\s*', response)
    chinese_steps = re.findall(r'第([一二三四五六七八九十]+)步', response)
    bullet_steps = re.findall(r'(?:^|\n)\s*[-*]\s+', response)

    arabic_nums = sorted(set(int(s) for s in arabic_steps)) if arabic_steps else []
    chinese_nums = list(range(1, len(chinese_steps) + 1)) if chinese_steps else []

    if arabic_nums:
        style = "arabic"
        max_num = max(arabic_nums)
        # 检查连续性：1,2,3,...
        consistent = arabic_nums == list(range(arabic_nums[0], arabic_nums[0] + len(arabic_nums)))
    elif chinese_nums:
        style = "chinese"
        max_num = len(chinese_nums)
        consistent = True  # 中文编号难以检查连续性
    elif bullet_steps:
        style = "bullet"
        max_num = len(bullet_steps)
        consistent = True
    else:
        style = "none"
        max_num = 0
        consistent = False

    return {
        "has_numbering": style != "none",
        "numbering_style": style,
        "max_step_number": max_num,
        "numbering_consistent": consistent,
    }


def _check_logical_dependencies(response: str, expected_order: list) -> dict:
    """
    检查步骤之间的逻辑依赖关系。

    通过检查期望步骤关键词在回复中出现的顺序是否正确来判断。

    参数:
        response: 模型回复
        expected_order: 期望的关键词出现顺序列表（如 ["查看pom", "添加依赖", "配置"]）

    返回:
        dict: {
            "order_correct": bool,         # 出现顺序是否正确
            "found_in_order": list,        # 按出现位置排列的命中关键词
            "dependency_score": float,     # 逻辑依赖得分 0-1
        }
    """
    # 找到每个关键词在回复中的位置
    positions = {}
    for kw in expected_order:
        idx = response.find(kw)
        if idx >= 0:
            positions[kw] = idx

    found_keywords = [kw for kw in expected_order if kw in positions]
    found_in_order = sorted(found_keywords, key=lambda k: positions[k])

    if len(found_keywords) < 2:
        # 少于2个关键词无法判断顺序
        return {
            "order_correct": len(found_keywords) > 0,
            "found_in_order": found_in_order,
            "dependency_score": len(found_keywords) / max(len(expected_order), 1),
        }

    # 检查顺序是否与期望一致
    expected_indices = [expected_order.index(kw) for kw in found_in_order]
    order_correct = expected_indices == sorted(expected_indices)

    # 得分：覆盖率 × 顺序正确性加成
    coverage = len(found_keywords) / len(expected_order)
    order_bonus = 1.0 if order_correct else 0.7
    dependency_score = coverage * order_bonus

    return {
        "order_correct": order_correct,
        "found_in_order": found_in_order,
        "dependency_score": dependency_score,
    }


def test_task_planning(model, tokenizer, verbose=True):
    """
    测试任务规划能力，返回 (results, score_0_to_100)。

    评分维度（每个用例）：
    - 步骤规划存在性 (25%): 是否有明确的步骤规划迹象
    - 步骤编号清晰度 (20%): 是否有清晰连续的步骤编号
    - 工具调用 (20%): 是否通过工具执行了规划中的步骤
    - 步骤覆盖度 (15%): 期望步骤关键词的覆盖率
    - 逻辑依赖正确性 (20%): 步骤之间的逻辑顺序是否正确
    """
    tools = json.loads(TOOL_DEFINITIONS_STR)

    test_cases = [
        {
            "prompt": "帮我给Spring Boot项目添加Flyway数据库迁移支持，项目使用MySQL。",
            "expect_steps": ["查看pom", "添加依赖", "配置", "创建迁移脚本"],
            "expect_order": ["查看", "依赖", "配置", "迁移"],
            "description": "多步骤任务 - 添加Flyway",
        },
        {
            "prompt": "线上用户反馈创建订单接口返回500错误，帮我排查。",
            "expect_steps": ["查看日志", "定位代码", "分析原因"],
            "expect_order": ["日志", "代码", "原因"],
            "description": "多步骤任务 - 线上排查",
        },
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("测试3：任务规划能力")
        print("=" * 60)

    results = []
    for i, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\n[{i}/{len(test_cases)}] {tc['description']}")
            print(f"  问题: {tc['prompt']}")

        messages = [
            {"role": "system", "content": "你是一位智能编程助手，可以通过调用工具来帮助用户完成软件开发任务。处理复杂任务时，请先拆解为具体步骤，然后按顺序执行。"},
            {"role": "user", "content": tc["prompt"]},
        ]

        response = generate(messages, model, tokenizer, max_new_tokens=1500, tools=tools)

        # (1) 步骤规划存在性 (25%)
        has_planning = any(kw in response for kw in ["步骤", "首先", "1.", "1、", "第一步", "第一", "计划"])
        planning_score = 1.0 if has_planning else 0.0

        # (2) 步骤编号清晰度 (20%)
        numbering_info = _check_step_numbering(response)
        if numbering_info["has_numbering"]:
            numbering_score = 0.6
            if numbering_info["numbering_consistent"]:
                numbering_score += 0.2
            if numbering_info["max_step_number"] >= 3:
                numbering_score += 0.2  # 至少3个步骤
        else:
            numbering_score = 0.0

        # (3) 工具调用 (20%)
        has_tool_call = "<tool_call>" in response
        tool_score = 1.0 if has_tool_call else 0.0

        # (4) 步骤覆盖度 (15%)
        step_hits = sum(1 for step_kw in tc["expect_steps"]
                        if any(kw in response for kw in [step_kw]))
        step_coverage = step_hits / len(tc["expect_steps"]) if tc["expect_steps"] else 0

        # (5) 逻辑依赖正确性 (20%)
        dep_info = _check_logical_dependencies(response, tc["expect_order"])

        # 综合得分
        case_score = (
            planning_score * 0.25
            + numbering_score * 0.20
            + tool_score * 0.20
            + step_coverage * 0.15
            + dep_info["dependency_score"] * 0.20
        )

        scoring_breakdown = {
            "planning_score": round(planning_score, 3),
            "planning_weight": 0.25,
            "numbering_score": round(numbering_score, 3),
            "numbering_weight": 0.20,
            "numbering_detail": numbering_info,
            "tool_score": round(tool_score, 3),
            "tool_weight": 0.20,
            "step_coverage": round(step_coverage, 3),
            "step_coverage_weight": 0.15,
            "dependency_score": round(dep_info["dependency_score"], 3),
            "dependency_weight": 0.20,
            "dependency_detail": dep_info,
        }

        if verbose:
            print(f"  包含步骤规划: {'是' if has_planning else '否'}")
            print(f"  步骤编号: {numbering_info['numbering_style']}"
                  f" (最多{numbering_info['max_step_number']}步"
                  f"{', 连续' if numbering_info['numbering_consistent'] else ', 不连续'})")
            print(f"  包含工具调用: {'是' if has_tool_call else '否'}")
            print(f"  步骤覆盖度: {step_hits}/{len(tc['expect_steps'])}")
            print(f"  逻辑依赖: {'顺序正确' if dep_info['order_correct'] else '顺序有误'}"
                  f" (覆盖 {dep_info['dependency_score']:.0%})")
            print(f"  回复长度: {len(response)} 字符")
            print(f"  综合得分: {case_score:.1%}")

        results.append({
            "description": tc["description"],
            "prompt": tc["prompt"],
            "response": response[:800],
            "has_planning": has_planning,
            "has_tool_call": has_tool_call,
            "step_coverage": step_coverage,
            "case_score": round(case_score * 100, 1),
            "scoring_breakdown": scoring_breakdown,
        })

    dimension_score = round(sum(r["case_score"] for r in results) / len(results), 1)
    if verbose:
        print(f"\n任务规划维度得分: {dimension_score}/100")
    return results, dimension_score


def test_frontend_dev(model, tokenizer, verbose=True):
    """
    测试前端开发能力（占位函数）。

    完整实现将在 US-010 中添加，包含 8-10 个测试用例覆盖全部子类别。
    当前版本返回空结果和 0 分。

    返回:
        tuple: (results: list[dict], score: float)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("测试4：前端开发能力（占位）")
        print("=" * 60)
        print("  [占位] 完整评估将在后续迭代实现")
        print("  前端开发维度得分: 0.0/100")
    return [], 0.0


# ============================================================
# 统一评估入口
# ============================================================


def evaluate_model(model, tokenizer, model_label: str, verbose=True):
    """
    对模型运行全部 4 个维度的测试，返回结构化评估结果。

    返回:
        dict: {
            "model_label": str,
            "dimensions": {
                "java_coding": {"results": [...], "score": float},
                "tool_calling": {"results": [...], "score": float},
                "task_planning": {"results": [...], "score": float},
                "frontend_dev": {"results": [...], "score": float},
            },
            "overall_score": float,
        }
    """
    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# 评估模型: {model_label}")
        print(f"{'#' * 60}")

    java_results, java_score = test_java_coding(model, tokenizer, verbose=verbose)
    tool_results, tool_score = test_tool_calling(model, tokenizer, verbose=verbose)
    plan_results, plan_score = test_task_planning(model, tokenizer, verbose=verbose)
    frontend_results, frontend_score = test_frontend_dev(model, tokenizer, verbose=verbose)

    overall = round((java_score + tool_score + plan_score + frontend_score) / 4, 1)

    evaluation = {
        "model_label": model_label,
        "dimensions": {
            "java_coding": {"results": java_results, "score": java_score},
            "tool_calling": {"results": tool_results, "score": tool_score},
            "task_planning": {"results": plan_results, "score": plan_score},
            "frontend_dev": {"results": frontend_results, "score": frontend_score},
        },
        "overall_score": overall,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"评估汇总 [{model_label}]")
        print(f"{'=' * 60}")
        print(f"  Java编程:   {java_score}/100")
        print(f"  工具调用:   {tool_score}/100")
        print(f"  任务规划:   {plan_score}/100")
        print(f"  前端开发:   {frontend_score}/100")
        print(f"  综合得分:   {overall}/100")

    return evaluation


# ============================================================
# 对比报告生成
# ============================================================


def _pick_example(base_results, finetuned_results, dimension: str):
    """
    从一个维度中选出最能体现改进的一个测试用例，返回对比样例。
    选择标准：微调模型得分 > 原始模型得分 差距最大的用例；
    如果没有改进的用例，选第一个。

    所有维度现在都使用 "score" 字段（0-1 范围），
    task_planning 使用 "case_score" (0-100 范围，除以100)。
    """
    best_diff = -float("inf")
    best_idx = 0

    for idx in range(len(base_results)):
        b = base_results[idx]
        f = finetuned_results[idx]

        if dimension == "task_planning":
            b_score = b.get("case_score", 0) / 100.0
            f_score = f.get("case_score", 0) / 100.0
        else:
            # java_coding, tool_calling, frontend_dev 均使用 score (0-1)
            b_score = b.get("score", 0)
            f_score = f.get("score", 0)

        diff = f_score - b_score
        if diff > best_diff:
            best_diff = diff
            best_idx = idx

    b = base_results[best_idx]
    f = finetuned_results[best_idx]

    # 截断回复以避免报告过长
    max_resp_len = 500
    return {
        "prompt": b.get("prompt", ""),
        "description": b.get("description", ""),
        "base_response": (b.get("response", ""))[:max_resp_len],
        "finetuned_response": (f.get("response", ""))[:max_resp_len],
    }


def generate_comparison_report(base_eval: dict, finetuned_eval: dict):
    """
    生成对比报告数据结构。

    返回:
        dict: 包含 per-dimension 对比、改进率、具体示例
    """
    dimension_names = {
        "java_coding": "Java编程",
        "tool_calling": "工具调用",
        "task_planning": "任务规划",
        "frontend_dev": "前端开发",
    }

    comparisons = {}
    for dim_key, dim_cn in dimension_names.items():
        base_score = base_eval["dimensions"][dim_key]["score"]
        ft_score = finetuned_eval["dimensions"][dim_key]["score"]

        if base_score > 0:
            improvement_pct = round((ft_score - base_score) / base_score * 100, 1)
        else:
            improvement_pct = round(ft_score, 1) if ft_score > 0 else 0.0

        example = _pick_example(
            base_eval["dimensions"][dim_key]["results"],
            finetuned_eval["dimensions"][dim_key]["results"],
            dim_key,
        )

        comparisons[dim_key] = {
            "dimension_cn": dim_cn,
            "base_score": base_score,
            "finetuned_score": ft_score,
            "improvement_pct": improvement_pct,
            "improved": ft_score > base_score,
            "example": example,
        }

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": base_eval["model_label"],
        "finetuned_model": finetuned_eval["model_label"],
        "base_overall": base_eval["overall_score"],
        "finetuned_overall": finetuned_eval["overall_score"],
        "overall_improvement_pct": round(
            (finetuned_eval["overall_score"] - base_eval["overall_score"])
            / max(base_eval["overall_score"], 0.01) * 100, 1
        ),
        "comparisons": comparisons,
    }
    return report


def write_comparison_json(report: dict, output_path: str):
    """将对比报告写入 JSON 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"对比报告 (JSON) 已保存到: {output_path}")


def write_comparison_markdown(report: dict, output_path: str):
    """将对比报告写入 Markdown 文件"""
    lines = []
    lines.append("# 模型对比评估报告\n")
    lines.append(f"**生成时间:** {report['timestamp']}\n")
    lines.append(f"**原始模型:** `{report['base_model']}`\n")
    lines.append(f"**微调模型:** `{report['finetuned_model']}`\n")
    lines.append("")

    # 总览表格
    lines.append("## 评分总览\n")
    lines.append("| 维度 | 原始模型 | 微调模型 | 改进幅度 |")
    lines.append("|------|---------|---------|---------|")

    for dim_key in ["java_coding", "tool_calling", "task_planning", "frontend_dev"]:
        c = report["comparisons"][dim_key]
        arrow = "+" if c["improvement_pct"] > 0 else ""
        lines.append(
            f"| {c['dimension_cn']} | {c['base_score']:.1f} | {c['finetuned_score']:.1f} | {arrow}{c['improvement_pct']:.1f}% |"
        )

    overall_arrow = "+" if report["overall_improvement_pct"] > 0 else ""
    lines.append(
        f"| **综合** | **{report['base_overall']:.1f}** | **{report['finetuned_overall']:.1f}** | **{overall_arrow}{report['overall_improvement_pct']:.1f}%** |"
    )
    lines.append("")

    # 每个维度的具体示例
    lines.append("## 具体示例对比\n")

    for dim_key in ["java_coding", "tool_calling", "task_planning", "frontend_dev"]:
        c = report["comparisons"][dim_key]
        ex = c["example"]
        status = "提升" if c["improved"] else ("持平" if c["improvement_pct"] == 0 else "下降")

        lines.append(f"### {c['dimension_cn']} ({status})\n")
        lines.append(f"**测试用例:** {ex['description']}\n")
        lines.append(f"**问题:** {ex['prompt'][:200]}\n")
        lines.append("")
        lines.append("**原始模型回复:**")
        lines.append(f"```\n{ex['base_response']}\n```\n")
        lines.append("**微调模型回复:**")
        lines.append(f"```\n{ex['finetuned_response']}\n```\n")
        lines.append("---\n")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"对比报告 (Markdown) 已保存到: {output_path}")


# ============================================================
# CLI 和主函数
# ============================================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="测试模型微调效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 测试 LoRA 适配器（自动加载基础模型）
  python test_model.py --model lora_adapter

  # 测试合并后的完整模型
  python test_model.py --model merged_model

  # 测试原始基础模型
  python test_model.py --model /path/to/base/model

    # 使用默认路径（从 config/config.yaml 读取 output.adapter_dir）
  python test_model.py

  # 对比模式：原始模型 vs 微调模型
  python test_model.py --compare

  # 对比模式：指定微调模型路径
  python test_model.py --compare --model lora_adapter
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径：合并模型目录 或 LoRA 适配器目录（默认从 config/config.yaml 读取 output.adapter_dir）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比模式：依次加载原始模型和微调模型，在相同测试集上评估并输出对比报告",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="测试结果输出文件路径（默认 test_results.json）",
    )
    return parser.parse_args()


def resolve_model_path(args, cfg):
    """根据 CLI 参数和配置文件确定微调模型路径"""
    if args.model:
        return args.model

    adapter_dir = get(cfg, "output.adapter_dir", "lora_adapter")
    merged_dir = get(cfg, "output.merged_dir", "merged_model")

    if Path(adapter_dir).exists():
        return adapter_dir
    elif Path(merged_dir).exists():
        return merged_dir
    else:
        base_path = get(cfg, "model.path", "")
        print(f"注意: 未找到适配器或合并模型，使用基础模型: {base_path}")
        return base_path


def run_single_mode(args, cfg):
    """单模型测试模式"""
    model_path = resolve_model_path(args, cfg)
    base_model_path = get(cfg, "model.path", "")

    print("=" * 60)
    print("模型测试")
    print(f"  模型路径: {model_path}")
    if is_lora_adapter(model_path):
        print(f"  模式: LoRA 适配器（基础模型 + 适配器）")
    else:
        print(f"  模式: 完整模型")
    print(f"  输出文件: {args.output}")
    print("=" * 60)

    model, tokenizer = load_model(model_path, base_model_path=base_model_path)
    evaluation = evaluate_model(model, tokenizer, model_label=model_path)

    # 保存结果（兼容旧格式）
    output_data = {}
    for dim_key, dim_data in evaluation["dimensions"].items():
        output_data[dim_key] = dim_data["results"]
    output_data["_metadata"] = {
        "model_path": model_path,
        "is_lora_adapter": is_lora_adapter(model_path) if Path(model_path).exists() else False,
        "base_model_path": base_model_path if is_lora_adapter(model_path) and Path(model_path).exists() else None,
        "scores": {k: v["score"] for k, v in evaluation["dimensions"].items()},
        "overall_score": evaluation["overall_score"],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 {args.output}")

    release_model(model, tokenizer)


def run_compare_mode(args, cfg):
    """对比模式：依次评估原始模型和微调模型"""
    base_model_path = get(cfg, "model.path", "")
    finetuned_path = resolve_model_path(args, cfg)

    # 检查微调模型路径是否与基础模型相同（无意义的对比）
    base_resolved = str(Path(base_model_path).resolve())
    ft_resolved = str(Path(finetuned_path).resolve())
    if base_resolved == ft_resolved:
        print("警告: 微调模型路径与基础模型路径相同，对比模式无意义。")
        print(f"  基础模型: {base_model_path}")
        print(f"  微调模型: {finetuned_path}")
        print("请通过 --model 指定微调模型路径，或确保 LoRA 适配器/合并模型已生成。")
        sys.exit(1)

    print("=" * 60)
    print("模型对比评估")
    print(f"  原始模型: {base_model_path}")
    print(f"  微调模型: {finetuned_path}")
    print("=" * 60)
    print("\n注意: 对比模式下将依次加载两个模型，每次只有一个模型在 GPU 上。")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB" if torch.cuda.is_available() else "  GPU: 不可用")
    print()

    # === 阶段1：评估原始模型 ===
    print("=" * 60)
    print("阶段 1/2：评估原始模型")
    print("=" * 60)
    model, tokenizer = load_model(base_model_path)
    base_eval = evaluate_model(model, tokenizer, model_label=base_model_path)
    release_model(model, tokenizer)
    print("\n原始模型评估完成，已释放 GPU 显存。\n")

    # === 阶段2：评估微调模型 ===
    print("=" * 60)
    print("阶段 2/2：评估微调模型")
    print("=" * 60)
    model, tokenizer = load_model(finetuned_path, base_model_path=base_model_path)
    ft_eval = evaluate_model(model, tokenizer, model_label=finetuned_path)
    release_model(model, tokenizer)
    print("\n微调模型评估完成，已释放 GPU 显存。\n")

    # === 生成对比报告 ===
    report = generate_comparison_report(base_eval, ft_eval)

    # 输出到控制台
    print("=" * 60)
    print("对比评估结果")
    print("=" * 60)
    for dim_key in ["java_coding", "tool_calling", "task_planning", "frontend_dev"]:
        c = report["comparisons"][dim_key]
        arrow = "+" if c["improvement_pct"] > 0 else ""
        status = "提升" if c["improved"] else ("持平" if c["improvement_pct"] == 0 else "下降")
        print(f"  {c['dimension_cn']:8s}: {c['base_score']:5.1f} -> {c['finetuned_score']:5.1f}  ({arrow}{c['improvement_pct']:.1f}%)  [{status}]")

    overall_arrow = "+" if report["overall_improvement_pct"] > 0 else ""
    print(f"  {'综合':8s}: {report['base_overall']:5.1f} -> {report['finetuned_overall']:5.1f}  ({overall_arrow}{report['overall_improvement_pct']:.1f}%)")

    # 保存 JSON 和 Markdown 报告
    json_path = "test_results_compare.json"
    md_path = "test_report.md"
    write_comparison_json(report, json_path)
    write_comparison_markdown(report, md_path)

    # 同时保存完整的原始评估数据（便于后续分析）
    full_data = {
        "base_evaluation": base_eval,
        "finetuned_evaluation": ft_eval,
        "comparison_report": report,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"完整评估数据已保存到: {args.output}")


def main():
    args = parse_args()

    # 加载配置（不传 CLI 参数，避免与 argparse 冲突）
    cfg = load_config(cli_args=[])

    if args.compare:
        run_compare_mode(args, cfg)
    else:
        run_single_mode(args, cfg)


if __name__ == "__main__":
    main()
