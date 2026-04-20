# PRD: Tool Calling 训练数据优化

## Introduction

当前 `data/categories/tool_calling.yaml` 及其配套的 seed 样本、工具定义存在与 opencode 实际工具体系严重脱节的问题。模型在复杂任务中表现出工具调用不积极、参数不合规、缺乏任务管理能力等缺陷。

本次优化通过三个层面的改动——扩展工具定义、增强 meta_prompt 和 subcategory、补充高质量 seed 样本——让训练数据准确反映 opencode 的工具使用规范，使微调后的模型能够连续稳定地执行复杂的多步骤任务。

## Goals

- 将工具定义从 6 个简化工具扩展到 10 个，对齐 opencode 实际工具体系（read_file, write_file, edit_file, search_files, find_files, run_command, list_dir, todo_write, ask_user, launch_agent）
- 每个工具定义包含完整的参数约束和使用规范（何时用、何时不用、关键限制）
- 新增 3 个 subcategory 覆盖积极调用、并行调用、任务管理等关键行为模式
- 增强 meta_prompt 中的 system prompt，内嵌 7 条工具使用策略
- 补充 8 条高质量 seed 样本覆盖新 subcategory（Java/Spring Boot 场景为主）
- 支持 YAML 级别的工具定义覆盖，使各 category 可独立配置工具列表

## User Stories

### US-001: 修改 generate_data.py 支持 YAML 级别工具定义覆盖

**Description:** 作为训练数据生成流程，我需要支持各 category YAML 独立配置工具定义，以便 tool_calling 可以使用完整版工具列表，而 frontend_dev 继续使用全局默认的精简版。

**Acceptance Criteria:**
- [ ] `_build_prompt()` 函数（约第 1461-1465 行）中，`tool_definitions` 的取值逻辑改为：优先从 `category_def.get("tool_definitions")` 读取，缺失时 fallback 到全局 `TOOL_DEFINITIONS_TEXT`
- [ ] 具体改动：`"tool_definitions": TOOL_DEFINITIONS_TEXT` → `"tool_definitions": category_def.get("tool_definitions", TOOL_DEFINITIONS_TEXT)`
- [ ] `frontend_dev.yaml` 和 `java_coding.yaml` 无需修改，不包含 `tool_definitions` 字段时自动使用全局默认值
- [ ] 运行 `python generate_data.py --dry-run` 或手动验证：tool_calling 使用 YAML 中的完整工具定义，frontend_dev 使用全局默认值

### US-002: 重写 tool_calling.yaml

**Description:** 作为训练数据生成流程，我需要一个全面升级的 tool_calling.yaml，包含扩展的工具定义、新增的 subcategory 和增强的 meta_prompt，以生成更高质量的工具调用训练样本。

**Acceptance Criteria:**
- [ ] 新增 `tool_definitions` 字段（YAML 多行字符串），包含 10 个工具的完整定义
- [ ] 每个工具定义包含：函数签名（标注必填/可选参数及类型）、功能描述、使用规范（⚠️ 标记关键约束）
- [ ] subcategories 从 5 个扩展到 8 个：保留原有 5 个 + 新增 `proactive_tool_use`、`parallel_tool_calls`、`task_management`
- [ ] meta_prompt 中的 system prompt 从一句话扩展为包含 7 条工具使用策略的完整指令
- [ ] meta_prompt 新增"核心行为模式"章节，明确要求样本体现 5 个行为（积极调用、参数精确、验证闭环、错误处理、专用工具优先）
- [ ] 为 3 个新 subcategory 编写 extra_instructions，每个包含关键行为清单
- [ ] YAML 语法正确，`python -c "import yaml; yaml.safe_load(open('data/categories/tool_calling.yaml'))"` 不报错
- [ ] meta_prompt 中的 JSON 花括号正确双写转义（`{{` 和 `}}`），Python `.format()` 渲染不报错

以下是 10 个工具定义的详细规格：

| # | 工具名 | 参数 | 关键约束 |
|---|--------|------|---------|
| 1 | `read_file` | `path: str`（必填）, `offset?: int`, `limit?: int` | path 必须绝对路径；默认最多 2000 行；编辑前必须先读取 |
| 2 | `write_file` | `path: str`（必填）, `content: str`（必填） | path 必须绝对路径；已有文件必须先 read_file；优先用 edit_file |
| 3 | `edit_file` | `path: str`（必填）, `old_text: str`（必填）, `new_text: str`（必填）, `replace_all?: bool` | old_text 必须完全匹配含缩进；多处匹配时需加上下文；使用前必须先 read_file |
| 4 | `search_files` | `pattern: str`（必填）, `path?: str`, `include?: str` | 正则语法；include 按文件名过滤；结果最多 100 条 |
| 5 | `find_files` | `pattern: str`（必填）, `path?: str` | glob 模式；按修改时间排序；结果最多 100 条 |
| 6 | `run_command` | `command: str`（必填）, `workdir?: str`, `timeout?: int`, `description: str`（必填） | 仅用于终端操作；不用于文件读写搜索；避免 cd && cmd；路径含空格必须引号；默认超时 120s |
| 7 | `list_dir` | `path?: str` | 自动忽略 node_modules/.git/dist 等；优先用 find_files/search_files |
| 8 | `todo_write` | `todos: list`（必填，每项含 content/status/priority） | 3+ 步骤任务必用；同时仅 1 个 in_progress；完成立即标记 completed |
| 9 | `ask_user` | `questions: list`（必填，每项含 question/header/options） | 推荐选项放首位加"(推荐)"；不包含"其他"选项 |
| 10 | `launch_agent` | `description: str`（必填）, `prompt: str`（必填）, `agent_type: str`（必填："explore"/"general"） | 仅用于开放性代码库探索；已知文件用 read_file；2-3 个文件用 search_files |

以下是 8 个 subcategory 的 extra_instructions 详细规格：

| subcategory | extra_instruction 内容 |
|-------------|----------------------|
| `simple_tool_call` | 简单工具调用：用户提出一个简单任务，assistant 调用 1 个工具即可完成。 |
| `multi_step_planning` | 多步骤任务规划：用户提出复杂任务，assistant 需要拆解为多个步骤，依次调用多个工具完成。每完成一步要总结结果并明确说明下一步。 |
| `tool_constraint` | 工具约束遵守：重点展示 assistant 严格遵守工具参数约束。包括：edit_file 的 old_text 精确匹配、run_command 的 workdir 使用、path 绝对路径、编辑前先读取等。 |
| `error_recovery` | 错误恢复链路：工具执行出错后，assistant 分析具体错误信息、定位根因、调整参数或策略后重试，直到成功。不能忽略错误继续执行。 |
| `refusal_or_fallback` | 拒绝或降级：用户的问题不需要工具调用（纯知识问答）时直接回答不调用工具；或者用户的操作任务需要通过工具执行（如安装软件、运行命令）。 |
| `proactive_tool_use` | 积极主动调用工具：面对需要信息的任务时，assistant 不猜测不假设，立即调用工具获取真实信息。关键行为：编辑文件前先 read_file 查看当前内容；不确定文件位置时先 find_files 或 search_files 定位；执行完操作后再调用工具验证结果；当一个工具返回的信息不够时，继续调用更多工具补充上下文。 |
| `parallel_tool_calls` | 并行工具调用：当多个工具调用之间没有依赖关系时，assistant 在同一条消息中同时发起多个工具调用（包含多个 `<tool_call>` 块）。关键行为：需要读取多个独立文件时同时调用多个 read_file；需要搜索多种模式时同时发起多个 search_files；有依赖关系的调用必须串行（如先读取再编辑）。 |
| `task_management` | 任务管理：面对复杂任务（3+ 步骤）时，assistant 必须使用 todo_write 工具进行任务规划和进度跟踪。关键行为：收到复杂任务后先用 todo_write 创建任务列表；开始某个步骤时标记 in_progress；完成后立即标记 completed；发现新的子任务时更新列表；同一时间只有 1 个 in_progress。 |

以下是增强后 meta_prompt 中 system prompt 的 7 条工具使用策略：

1. 需要信息时立即调用工具，不要猜测
2. 编辑文件前必须先用 read_file 查看内容
3. 文件操作优先使用专用工具（read_file/edit_file/search_files），不要用 run_command 替代
4. 多个独立的工具调用可以并行发起（在同一条消息中包含多个 tool_call）
5. 复杂任务先用 todo_write 制定计划，逐步执行
6. 避免 cd && command 模式，使用 workdir 参数
7. 命令执行后检查结果，出错时分析原因并修复

以下是增强后 meta_prompt 中"核心行为模式"章节的 5 个行为：

1. **积极调用** — assistant 不做假设，遇到不确定的信息就调用工具确认
2. **参数精确** — 工具参数严格遵守定义中的约束（绝对路径、必填参数等）
3. **验证闭环** — 执行修改后调用工具验证结果
4. **错误不放过** — 工具返回错误时分析原因，调整参数重试
5. **专用工具优先** — 文件读写用 read_file/edit_file，不用 run_command 中的 cat/sed

### US-003: 补充 tool_calling seed 样本

**Description:** 作为训练数据生成流程，我需要高质量的 seed 样本覆盖新增的 3 个 subcategory 以及强化原有 subcategory 的典型场景，以确保 LLM 生成的训练数据质量。

**Acceptance Criteria:**
- [ ] 在 `data/seeds/tool_calling.jsonl` 末尾追加 8 条新样本（第 11-18 条）
- [ ] 新样本的 subcategory/difficulty 分布：proactive_tool_use ×2, parallel_tool_calls ×2, task_management ×2, tool_constraint ×1, multi_step_planning ×1
- [ ] 所有样本以 Java/Spring Boot 为主要场景
- [ ] 每条样本的 system prompt 使用增强后的版本（包含工具使用策略）
- [ ] 所有工具调用使用 CoPaw XML 格式（`<tool_call>...<function=name>...<parameter=name>...</tool_call>`）
- [ ] 工具参数严格遵守 US-002 中定义的约束（绝对路径、description 必填等）
- [ ] 并行调用样本中，单条 assistant 消息包含多个 `<tool_call>` 块，对应的 user 消息包含多个 `<tool_response>` 块
- [ ] 任务管理样本中，assistant 在每个阶段调用 `todo_write` 更新状态（pending → in_progress → completed）
- [ ] proactive_tool_use 样本展示"不猜测 → 先查看 → 再行动 → 后验证"的完整链路
- [ ] 每条样本包含正确的 metadata：`{"category": "tool_calling", "subcategory": "...", "difficulty": "..."}`
- [ ] 每条样本是合法的单行 JSON，`python -c "import json; [json.loads(l) for l in open('data/seeds/tool_calling.jsonl')]"` 不报错

以下是 8 条 seed 样本的详细规格：

| # | subcategory | difficulty | 场景 | 核心示范行为 |
|---|------------|-----------|------|-------------|
| 11 | `proactive_tool_use` | intermediate | "帮我重构 OrderService 的 create 方法，把参数校验逻辑提取出来" | 不猜测代码 → 先 read_file 查看 → search_files 找引用 → edit_file 重构 → run_command(mvn compile) 验证 |
| 12 | `proactive_tool_use` | advanced | "项目 mvn compile 报了 3 个错误，帮看看" | 立刻 run_command 执行编译 → 逐个 read_file 查看报错文件 → 不假设原因，每个都查源码后诊断 |
| 13 | `parallel_tool_calls` | intermediate | "帮我了解这个 Spring Boot 项目的整体结构" | 同时 3 个并行 tool_call：list_dir("/project") + find_files("**/*.java", "/project/src") + read_file("/project/pom.xml") |
| 14 | `parallel_tool_calls` | basic | "帮我对比 dev 和 prod 环境的数据库配置有什么不同" | 同时 2 个并行 read_file：application-dev.yml + application-prod.yml，然后逐项对比分析 |
| 15 | `task_management` | advanced | "给项目添加 Swagger API 文档、请求参数校验和全局异常处理" | 先 todo_write(3 个 pending 任务) → 逐个标记 in_progress → 执行 → 标记 completed → 下一个 |
| 16 | `task_management` | intermediate | "编译有 3 个错误，帮我全部修好" | todo_write 创建 3 个修复任务 → 逐个 in_progress → read_file+edit_file 修复 → completed → 最后 run_command 验证全部通过 |
| 17 | `tool_constraint` | advanced | "把 UserService 中的 findById 方法改成返回 Optional" | read_file → edit_file 首次 old_text 缩进不对导致失败 → 重新 read_file 确认精确内容 → 修正 old_text 重试成功 |
| 18 | `multi_step_planning` | advanced | "给 UserController 加一个带分页和条件筛选的查询接口" | find_files 定位 → read_file(Controller+Service+Mapper) → edit_file(Mapper 加 SQL) → edit_file(Service) → edit_file(Controller) → run_command(mvn compile) 验证 |

## Functional Requirements

### 工具定义系统

- FR-1: `generate_data.py` 的 `_build_prompt()` 函数必须支持从 `category_def` dict 中读取 `tool_definitions` 字段，缺失时 fallback 到全局 `TOOL_DEFINITIONS_TEXT` 常量
- FR-2: `tool_calling.yaml` 的 `tool_definitions` 字段必须包含 10 个工具的完整定义，每个工具包含函数签名、功能描述和使用规范
- FR-3: 工具定义中的参数约束必须与 opencode 源码中的实际实现一致（参考 `opencode/packages/opencode/src/tool/` 下各工具的 `.txt` 描述文件）
- FR-4: 全局 `TOOL_DEFINITIONS_TEXT` 常量保持不变，作为其他 category 的 fallback

### Subcategory 系统

- FR-5: `subcategories` 列表包含 8 个条目，顺序为：simple_tool_call, multi_step_planning, tool_constraint, error_recovery, refusal_or_fallback, proactive_tool_use, parallel_tool_calls, task_management
- FR-6: `extra_instructions` 中为全部 8 个 subcategory 提供详细的行为指导

### Meta Prompt

- FR-7: system prompt 必须包含 7 条工具使用策略，使用有序列表格式
- FR-8: meta_prompt 必须包含"核心行为模式"章节，列出 5 个必须在样本中体现的行为
- FR-9: 格式要求第 6 条从"总结结果"改为"总结结果并决定下一步"，引导连续执行行为

### Seed 样本

- FR-10: 新增的 8 条 seed 样本中，parallel_tool_calls 样本的单条 assistant 消息必须包含 2 个及以上 `<tool_call>` 块
- FR-11: 新增的 8 条 seed 样本中，task_management 样本必须包含至少 2 次 `todo_write` 工具调用（创建 + 至少一次状态更新）
- FR-12: 新增的 8 条 seed 样本中，proactive_tool_use 样本必须展示"先查后改"模式（edit_file 之前必有对应文件的 read_file）
- FR-13: 所有 seed 样本中的工具参数必须符合工具定义中的约束（如 path 使用绝对路径、run_command 包含 description 参数）

## Non-Goals (Out of Scope)

- **不修改 `frontend_dev.yaml`**：该文件继续使用全局默认工具定义，本次不涉及
- **不修改 `java_coding.yaml`**：该 category 不涉及工具调用，不受影响
- **不修改 `generate_data.py` 的验证逻辑**（`validate_sample`、`_validate_tool_calls`）：当前的 XML 格式验证已足够，新增工具名不影响格式验证
- **不修改 `generate_data.py` 的加载逻辑**（`load_categories`）：YAML 新增 `tool_definitions` 字段不在必填字段校验范围内，无需改动
- **不添加非 Java 语言的 seed 样本**：保持 Java/Spring Boot 为主
- **不添加 opencode 环境特有的工具**（`websearch`、`codesearch`、`lsp`、`skill`、`plan_exit`）：这些工具依赖特定环境/配置，不适合作为通用训练数据
- **不修改 CoPaw XML 工具调用格式**：保持现有格式不变
- **不修改训练脚本**（`train.py`）或评测脚本（`test_model.py`）

## Technical Considerations

### 与现有系统的兼容性

- `_build_prompt()` 的改动仅为一行代码的优先级调整（`category_def.get("tool_definitions", TOOL_DEFINITIONS_TEXT)`），向后完全兼容
- YAML 新增 `tool_definitions` 字段属于可选字段，`load_categories()` 的必填字段校验不涉及该字段，无需修改加载逻辑
- 新增 subcategory 对数据生成流程透明——`generate_via_llm()` 通过 round-robin 遍历 subcategories 列表，新增条目自动参与轮转

### YAML 模板技术约束

- `tool_definitions` 字段以 YAML 多行字符串（`|`）形式存储，在 `meta_prompt` 的 `{tool_definitions}` 占位符处注入
- `meta_prompt` 模板中所有 JSON 花括号必须双写转义（`{{` 和 `}}`），否则 Python `.format()` 会报 KeyError
- `tool_definitions` 字段本身**不需要**双写转义，因为它作为一个完整的字符串值被注入，不经过 `.format()` 处理

### Seed 样本技术约束

- 每条 seed 必须是单行合法 JSON（不能有换行符，除非在字符串值的 `\n` 转义中）
- `<tool_call>` 和 `<tool_response>` 标签必须配对闭合
- 并行工具调用样本中，assistant 消息的多个 `<tool_call>` 块之间用 `\n\n` 分隔
- 对应的 user 消息中包含多个 `<tool_response>` 块，按调用顺序排列

### opencode 工具映射关系

训练数据中的工具名 → opencode 实际工具名的映射：

| 训练数据工具名 | opencode 工具名 | 来源文件 |
|---------------|----------------|---------|
| `read_file` | `read` | `opencode/src/tool/read.ts` |
| `write_file` | `write` | `opencode/src/tool/write.ts` |
| `edit_file` | `edit` | `opencode/src/tool/edit.ts` |
| `search_files` | `grep` | `opencode/src/tool/grep.ts` |
| `find_files` | `glob` | `opencode/src/tool/glob.ts` |
| `run_command` | `bash` | `opencode/src/tool/bash.ts` |
| `list_dir` | `list` | `opencode/src/tool/ls.ts` |
| `todo_write` | `todowrite` | `opencode/src/tool/todo.ts` |
| `ask_user` | `question` | `opencode/src/tool/question.ts` |
| `launch_agent` | `task` | `opencode/src/tool/task.ts` |

训练数据使用语义化的工具名（如 `read_file` 而非 `read`），因为目标模型 CoPaw-Flash-9B 是通用工具调用模型，不绑定 opencode 的具体工具名。核心是让模型学会**行为模式和使用规范**。

## Success Metrics

- 生成的训练样本中，工具参数合规率（符合工具定义约束）从当前估计的 ~70% 提升到 90%+
- 生成的 multi_step_planning 样本中，平均工具调用次数从 ~3 次提升到 5+ 次
- 新增 subcategory（proactive/parallel/task_management）的样本成功生成率 > 80%
- `python generate_data.py --category tool_calling --count 10` 运行成功，生成的样本通过 `validate_sample()` 验证
- seed 样本总数从 10 条增加到 18 条，覆盖全部 8 个 subcategory

## Open Questions

1. **并行调用的 tool_response 格式**：当 assistant 在一条消息中发出多个 `<tool_call>` 时，对应的 user 消息中多个 `<tool_response>` 的分隔方式是用换行分隔还是其他方式？当前 seed 中没有并行调用的先例。建议用 `\n\n` 分隔多个 `<tool_response>` 块。
2. **todo_write 的 tool_response 格式**：todo_write 是状态管理工具，返回值应该是什么？建议返回 JSON 格式的 todo 列表快照。
3. **launch_agent 是否纳入 seed 样本**：launch_agent 对应 opencode 的 Task 工具（启动子代理），场景较特殊。当前计划不在 seed 中使用该工具，仅在工具定义中声明，让 LLM 生成时自行决定是否使用。如有需要可后续补充。
4. **工具定义文本长度**：扩展后的 tool_definitions 约 1500-2000 字符，注入 meta_prompt 后总长度会增加。需确认不超过 LLM API 的 max_tokens 限制（当前 `max_tokens = min(batch_size * 4096, 16384)`）。meta_prompt 属于 system/user prompt，不占 max_tokens（output tokens），但占上下文窗口。DeepSeek 的上下文窗口为 64K，应足够。
