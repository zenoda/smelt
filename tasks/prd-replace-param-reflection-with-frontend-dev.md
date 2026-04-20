# PRD: 删除 param_reflection 样本类型，新增 frontend_dev 前端开发样本类型

## Introduction

当前训练数据包含三种样本类型：`java_coding`(50%)、`tool_calling`(30%)、`param_reflection`(20%)。本次变更将删除 `param_reflection`（参数反思）类型，替换为 `frontend_dev`（前端开发）类型，使模型具备 Vue3 + TypeScript + ElementPlus 技术栈的前端编程能力。比例保持 50%/30%/20% 不变。

## Goals

- 完整移除 `param_reflection` 在所有活跃代码文件中的定义、配置、种子数据、评估逻辑和历史引用
- 新增 `frontend_dev` 类型，包含 YAML 定义、种子数据、评估函数
- 前端技术栈覆盖：JavaScript、TypeScript、Vue3（Composition API）、ElementPlus、axios、CSS/SCSS
- 训练比例：`java_coding: 0.50`、`tool_calling: 0.30`、`frontend_dev: 0.20`
- 所有现有功能（数据生成、合并、训练、评估）正常工作

## User Stories

### US-001: 创建 frontend_dev 类型定义文件

**Description:** 作为数据生成流程，我需要一个 `data/categories/frontend_dev.yaml` 定义文件，以便 `load_categories()` 能自动加载前端开发样本类型。

**Acceptance Criteria:**

- [ ] 创建 `data/categories/frontend_dev.yaml`，结构与 `java_coding.yaml` 一致
- [ ] `name` 字段为 `frontend_dev`
- [ ] `description` 字段为 "前端开发 — Vue3、ElementPlus、TypeScript、CSS、axios"
- [ ] 包含 8 个子类别（subcategories）：`vue3_basics`、`vue3_components`、`element_plus`、`typescript_vue`、`axios_http`、`css_layout`、`state_management`、`frontend_engineering`
- [ ] `meta_prompt` 模板包含正确的占位符（`{seed_example}`、`{difficulty_instruction}`、`{subcategory}`、`{tool_definitions}`、`{extra_instruction}`、`{batch_size}`），内容引导 LLM 生成前端开发训练样本
- [ ] `meta_prompt` 中的 system 角色提示词为："你是一位资深前端开发工程师，精通 Vue3、TypeScript、ElementPlus、axios、CSS 等技术栈。请用简洁准确的语言回答问题，代码示例要完整可运行。"
- [ ] `meta_prompt` 中的格式要求明确：代码示例使用 Vue3 Composition API（`<script setup lang="ts">`）、TypeScript 类型标注、ElementPlus 组件
- [ ] `extra_instructions` 为每个子类别提供详细的聚焦指令
- [ ] YAML 格式合法，可被 `generate_data.py` 中的 `load_categories()` 正确解析

### US-002: 创建 frontend_dev 种子数据

**Description:** 作为数据生成流程，我需要 `data/seeds/frontend_dev.jsonl` 种子文件，以便 LLM 生成时有参考样本。

**Acceptance Criteria:**

- [ ] 创建 `data/seeds/frontend_dev.jsonl`，每行一个 JSON 对象
- [ ] 包含 18-21 条手写种子样本，覆盖全部 8 个子类别，每个子类别至少 2 条
- [ ] 每条样本格式：`{"messages": [...], "metadata": {"category": "frontend_dev", "subcategory": "xxx", "difficulty": "basic|intermediate|advanced"}}`
- [ ] `messages` 数组第一条必须是 system role，内容为上述 system 提示词
- [ ] assistant 回复包含完整可运行的代码示例（使用 ````vue`、````typescript`、````css` 等代码块）
- [ ] 代码示例遵循 Vue3 Composition API + TypeScript 规范
- [ ] 难度分布合理：basic 约 30%、intermediate 约 50%、advanced 约 20%

子类别数量分配建议：

| 子类别 | 数量 | 难度 |
|--------|------|------|
| `vue3_basics` | 3 | basic/intermediate |
| `vue3_components` | 3 | intermediate/advanced |
| `element_plus` | 3 | basic/intermediate |
| `typescript_vue` | 2 | intermediate/advanced |
| `axios_http` | 2 | intermediate |
| `css_layout` | 2 | basic/intermediate |
| `state_management` | 2 | intermediate |
| `frontend_engineering` | 2 | advanced |

### US-003: 删除 param_reflection 类型定义和种子数据

**Description:** 作为维护者，我需要移除 `param_reflection` 的数据文件，避免遗留无用文件。

**Acceptance Criteria:**

- [ ] 删除 `data/categories/param_reflection.yaml`
- [ ] 删除 `data/seeds/param_reflection.jsonl`
- [ ] 确认 `data/categories/` 目录下只剩 `java_coding.yaml`、`tool_calling.yaml`、`frontend_dev.yaml`
- [ ] 确认 `data/seeds/` 目录下只剩 `java_coding.jsonl`、`tool_calling.jsonl`、`frontend_dev.jsonl`

### US-004: 更新配置文件中的类别比例

**Description:** 作为数据生成流程，我需要配置文件中的 `category_ratios` 反映新的类型列表和比例。

**Acceptance Criteria:**

- [ ] `config/config.yaml` 中 `data.category_ratios` 更新为：`java_coding: 0.50`、`tool_calling: 0.30`、`frontend_dev: 0.20`
- [ ] `config/config.example.yaml` 中同步更新
- [ ] 移除 `param_reflection` 相关的注释
- [ ] 新增 `frontend_dev` 的中文注释（`# 前端开发样本`）

### US-005: 更新 generate_data.py 中的硬编码默认比例

**Description:** 作为数据生成脚本，`generate_data.py` 中有两处硬编码的默认 `category_ratios`（`--merge` 模式和 LLM 生成模式），需要同步更新。

**Acceptance Criteria:**

- [ ] `--merge` 模式默认比例（约 L1952-1955）：`param_reflection: 0.20` 改为 `frontend_dev: 0.20`
- [ ] LLM 生成模式默认比例（约 L2022-2025）：同上
- [ ] `load_seeds()` 函数的文档注释（约 L174）中如有 `param_reflection` 示例，更新为 `frontend_dev`
- [ ] 全文搜索确认 `generate_data.py` 中不再有 `param_reflection` 字符串

### US-006: 更新 train.py 中的类别推断逻辑

**Description:** 作为训练流程，`train.py` 的 `_infer_category()` 函数需要能正确推断 `frontend_dev` 类型的样本。

**Acceptance Criteria:**

- [ ] 删除 `_infer_category()` 中 `param_reflection` 的关键词推断逻辑（约 L70-80 的两段 `param_keywords` 检测）
- [ ] 新增 `frontend_dev` 的内容推断逻辑：检测关键词如 "Vue"、"ElementPlus"、"Element Plus"、"前端"、"组件"、"Composition API"、"Pinia"、"Vite" 等
- [ ] 推断优先级：metadata.category（精确） > 工具调用检测 > 前端关键词检测 > 默认 java_coding
- [ ] 更新函数文档注释，反映新的推断策略

### US-007: 删除 test_model.py 中的 param_reflection 评估，新增 frontend_dev 评估

**Description:** 作为评估流程，需要用前端开发评估维度替换参数反思评估维度。

**Acceptance Criteria:**

- [ ] 删除 `test_param_reflection()` 函数（约 L857 起）
- [ ] 新增 `test_frontend_dev()` 函数，评分维度：
  - 代码正确性 (40%)：代码是否可运行、语法正确、逻辑合理
  - 最佳实践 (30%)：是否遵循 Vue3 Composition API、TypeScript 类型标注、组件化等最佳实践
  - 解释清晰度 (30%)：概念解释是否准确、代码注释是否充分
- [ ] 测试用例覆盖全部 8 个子类别，每个子类别至少 1 个用例，共 8-10 个
- [ ] 每个测试用例包含：`prompt`、`description`、`subcategory`、`check_fn`（检查回复中是否包含关键代码模式）、`practice_keywords`（最佳实践关键词）
- [ ] 返回格式与其他 test_xxx 函数一致：`(results_list, score_0_to_100)`
- [ ] 更新 `evaluate_model()` 函数（约 L1157-1183）：
  - 将 `param_results, param_score = test_param_reflection(...)` 替换为 `frontend_results, frontend_score = test_frontend_dev(...)`
  - 更新 `evaluation["dimensions"]` 中的 key
  - 更新 `overall` 计算公式
  - 更新 verbose 输出中的维度名称
- [ ] 更新 `dimension_names` 映射（约 L1242-1247）：`"frontend_dev": "前端开发"`
- [ ] 更新所有遍历维度 key 的循环（约 L1310、L1326 等）：`"param_reflection"` → `"frontend_dev"`
- [ ] 全文搜索确认 `test_model.py` 中不再有 `param_reflection` 字符串

### US-008: 更新 extract_seeds.py

**Description:** 作为种子提取工具，需要移除 `param_reflection` 的提取逻辑，适配新的 `frontend_dev` 种子来源。

**Acceptance Criteria:**

- [ ] 删除 `from generate_data import generate_param_reflection` 导入语句（L29）
- [ ] 删除 `extract_param_reflection()` 函数（L122-167）
- [ ] 删除 `main()` 中对 `extract_param_reflection()` 的调用和输出（L179, L187-188）
- [ ] 更新文件验证列表（L195）：`"param_reflection.jsonl"` → `"frontend_dev.jsonl"`
- [ ] 由于 frontend_dev 种子是直接手写 JSONL 而非从 generate_data.py 提取，在 `main()` 中添加对 `frontend_dev.jsonl` 的存在性检查和验证（格式校验），而非提取
- [ ] 脚本可正常运行，不报错

### US-009: 更新 README.md

**Description:** 作为项目文档，README 需要反映新的样本类型列表和比例。

**Acceptance Criteria:**

- [ ] 更新项目目录结构说明：`param_reflection.yaml` → `frontend_dev.yaml`，`param_reflection.jsonl` → `frontend_dev.jsonl`
- [ ] 更新样本类型比例表格：`param_reflection 20% 参数类型/缺失/合理性检查` → `frontend_dev 20% Vue3/ElementPlus/TypeScript/CSS 前端开发`
- [ ] 如有其他对 `param_reflection` 的描述性文字，同步更新

### US-010: 清理历史归档文件中的 param_reflection 引用

**Description:** 作为代码库维护，需要清理所有文件中对 `param_reflection` 的引用，包括历史归档。

**Acceptance Criteria:**

- [ ] 清理 `prd.json` 中的 `param_reflection` 引用
- [ ] 清理 `archive/2026-04-20-pluggable-sample-categories/prd.json` 中的 `param_reflection` 引用
- [ ] 清理 `tasks/prd-tool-calling-optimization.md` 中的 `param_reflection` 引用
- [ ] 全局搜索确认项目中不再有 `param_reflection` 字符串（git grep 验证）

## Functional Requirements

- FR-1: `data/categories/frontend_dev.yaml` 必须可被 `generate_data.py` 的 `load_categories()` 函数正确加载，包含 `name`、`description`、`subcategories`、`meta_prompt`、`extra_instructions` 字段
- FR-2: `data/seeds/frontend_dev.jsonl` 每行必须是合法 JSON，包含 `messages` 和 `metadata` 字段，`metadata` 包含 `category`、`subcategory`、`difficulty`
- FR-3: `meta_prompt` 模板中的占位符（`{seed_example}` 等）必须与 `generate_data.py` 的 `_build_prompt()` 函数期望的一致
- FR-4: `config/config.yaml` 和 `config/config.example.yaml` 中 `category_ratios` 的值总和必须为 1.0
- FR-5: `train.py` 的 `_infer_category()` 必须能将含有 Vue/前端关键词的样本正确推断为 `frontend_dev`
- FR-6: `test_model.py` 的 `test_frontend_dev()` 返回值格式必须为 `(list[dict], float)`，其中 float 为 0-100 的分数，与其他 `test_xxx` 函数一致
- FR-7: `test_frontend_dev()` 的每个测试用例必须包含 `prompt`、`description`、`subcategory`、`check_fn` 字段
- FR-8: `evaluate_model()` 的 `overall` 分数必须是四个维度（java_coding、tool_calling、task_planning、frontend_dev）的平均值
- FR-9: 对比报告生成（`generate_comparison_report`、`write_comparison_markdown`）必须正确处理 `frontend_dev` 维度
- FR-10: 全局不残留 `param_reflection` 字符串（归档文件同步清理）

## Non-Goals

- 不修改 `tool_calling` 和 `java_coding` 类型的任何定义、种子数据或评估逻辑
- 不修改数据生成流程的核心架构（`load_categories()` 动态加载机制不变）
- 不修改训练流程（`train.py` 中除 `_infer_category()` 外的逻辑不变）
- 不新增 generate_data.py 中的手写样本生成函数（种子数据直接手写 JSONL）
- 不涉及模型训练和部署
- 不调整 `data/generated/` 目录下已生成的数据（如需重新生成，由用户手动触发）

## Technical Considerations

- **可插拔架构**：项目已实现可插拔样本类型系统，新增类型只需在 `data/categories/` 下创建 YAML 文件。本次变更主要工作在种子数据编写和评估函数实现上
- **meta_prompt 占位符**：必须与 `generate_data.py` 中 `_build_prompt()` 函数的模板渲染逻辑一致，特别是 `{tool_definitions}` 占位符 — 前端开发类型通常不需要 tool_definitions，但模板中仍需保留占位符以兼容通用渲染流程
- **评估函数设计**：前端代码的正确性难以像工具调用那样做精确的对错判断，`test_frontend_dev()` 应以关键词/模式匹配为主（如检测 `<script setup`、`defineProps`、`ref(`、`ElTable` 等），辅以启发式规则
- **extract_seeds.py 兼容性**：由于 frontend_dev 种子不从 generate_data.py 提取，extract_seeds.py 只需对已有的 `frontend_dev.jsonl` 做格式验证
- **generate_data.py 中可能存在的其他硬编码**：需全文搜索 `param_reflection` 确保无遗漏，注意检查注释和文档字符串

## Success Metrics

- `python generate_data.py --merge` 能正确合并三种新类型的数据，比例符合配置
- `python generate_data.py --category frontend_dev` 能正确生成前端开发样本（需配置 LLM API Key）
- `python train.py` 能正确加载含 `frontend_dev` 类型的训练数据，`_infer_category()` 推断准确
- `python test_model.py` 能完整运行四维评估（java_coding、tool_calling、task_planning、frontend_dev），输出评分报告
- `python extract_seeds.py` 能正常运行，验证所有种子文件格式正确
- `git grep param_reflection` 返回空结果（零残留）

## Open Questions

- `test_frontend_dev()` 的 `check_fn` 如何有效判断 Vue3 代码的质量？纯关键词匹配可能不够准确，是否需要引入更复杂的启发式规则（如检测 import 语句、组件结构完整性）？
- 前端开发样本的 system 提示词是否需要包含工具调用相关说明？当前 `java_coding` 的 system 提示词不包含工具调用，但 `param_reflection` 的包含。`frontend_dev` 应沿用 `java_coding` 的纯知识问答模式。
- 如果后续需要恢复"参数反思"能力，是否将其融入 `tool_calling` 类型的子类别中？本次 PRD 不处理这个问题，但值得后续讨论。
