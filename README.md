# QLoRA Training Pipeline

通用 QLoRA 微调训练流水线，支持任意兼容 Unsloth 的语言模型（如 Qwen、LLaMA、Mistral 等）。当前训练目标为提升模型在以下方面的能力：

1. **Java 编程** — Spring Boot/Cloud、MyBatis、Flyway、WebFlux、异常诊断
2. **任务拆解与工具调用** — 步骤规划、严格的 tool_call XML 格式遵循
3. **参数反思** — 参数类型检查、缺失检测、值合理性判断

## 硬件要求

| 资源 | 最低要求 | 说明 |
|------|---------|------|
| GPU | Nvidia RTX 2080 Ti (22GB VRAM) | QLoRA 4-bit 训练占用 ~10-12GB |
| RAM | 64GB | 模型合并时峰值 ~25-30GB |
| 磁盘 | 30GB+ | LoRA 适配器 ~250MB，合并模型 ~18GB |

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 创建配置文件
cp config/config.example.yaml config/config.yaml
# 编辑 config/config.yaml 中的模型路径等参数

# 3. 检查环境
make check

# 4. 运行完整流水线（数据生成 -> 训练 -> 评估）
make all
```

或分步执行：

```bash
make data       # 生成训练数据
make train      # QLoRA 训练
make test       # 模型评估
```

## 项目结构

```
.
├── Makefile                 # 流水线入口（make help 查看所有目标）
├── config/
│   ├── config.example.yaml  # 配置模板（所有参数含中文注释）
│   └── config.yaml          # 运行时配置（gitignored）
├── config_utils.py          # 配置加载工具（YAML + CLI 覆盖）
├── generate_data.py         # 训练数据生成（LLM API + 种子扩充）
├── train.py                 # QLoRA 训练脚本
├── test_model.py            # 模型评估脚本（4 维度打分 + 对比报告）
├── merge_lora.py            # 多 LoRA 适配器合并
├── manage_experiments.py    # 实验管理（列表 + 对比）
├── extract_seeds.py         # 种子样本提取工具（一次性使用）
├── requirements.txt         # Python 依赖
├── data/
│   ├── categories/          # 样本类型 YAML 定义（可插拔）
│   │   ├── java_coding.yaml        # Java 编程类型定义
│   │   ├── tool_calling.yaml       # 工具调用类型定义
│   │   └── param_reflection.yaml   # 参数反思类型定义
│   ├── seeds/               # 手写种子样本（Git 跟踪）
│   │   ├── java_coding.jsonl       # 21 条 Java 编程样本
│   │   ├── tool_calling.jsonl      # 10 条工具调用样本
│   │   └── param_reflection.jsonl  # 21 条参数反思样本
│   └── generated/           # LLM 生成的数据（gitignored）
├── training_data.jsonl      # 合并后的训练数据（gitignored）
├── lora_adapter/             # LoRA 适配器输出（gitignored）
├── merged_model/             # 合并模型输出（gitignored）
├── runs/                    # TensorBoard 日志（gitignored）
└── experiments/             # 实验记录（gitignored）
```

## Makefile 目标

| 目标 | 说明 |
|------|------|
| `make help` | 显示帮助信息（默认目标） |
| `make data` | 生成训练数据并合并 |
| `make train` | 运行 QLoRA 训练 |
| `make test` | 模型评估（自动检测适配器启用对比模式） |
| `make all` | 完整流水线：data -> train -> test |
| `make clean` | 清理生成文件（带确认提示） |
| `make tensorboard` | 启动 TensorBoard 服务 |
| `make check` | 环境检查报告 |
| `make merge` | 合并 LoRA 适配器到基础模型 |
| `make merge-lora` | 合并多个 LoRA 适配器 |

每个目标执行前会自动检查前置条件（venv、训练数据、适配器等）。

## 配置系统

配置文件 `config/config.yaml` 包含以下段落，每个参数均有详细中文注释：

| 段落 | 关键参数 | 说明 |
|------|---------|------|
| `model` | `path`, `load_in_4bit` | 基础模型路径和量化设置 |
| `lora` | `r` (32), `alpha` (32), `target_modules` | LoRA 适配器参数 |
| `training` | `learning_rate` (5e-5), `num_epochs` (3), `max_seq_length` (4096) | 训练超参数 |
| `data` | `sample_count`, `category_ratios`, `batch_size`, `categories_dir` | 数据生成与划分 |
| `output` | `adapter_dir`, `merged_dir`, `merge_model` | 输出路径与合并开关 |
| `early_stopping` | `enabled`, `patience` (3), `min_delta` | 早停配置 |
| `lora_merge` | `method`, `adapters`, `weights`, `density` | 多 LoRA 合并配置 |

CLI 参数可覆盖任意配置项，使用点号分隔路径：

```bash
python train.py --training.learning_rate 3e-5 --lora.r 16 --training.num_epochs 5
```

## 流水线详解

### 1. 数据生成 (`generate_data.py`)

支持两种模式：

- **LLM API 批量生成**：调用外部 LLM 批量生成训练数据变体（每次 API 调用生成多条样本）
- **种子扩充**：无 API Key 时回退到种子样本重复打散

```bash
# LLM 生成模式（需设置 API Key）
export LLM_API_KEY="your-api-key"
python generate_data.py --count 2000 --category java_coding

# 指定多个类别（逗号分隔）
python generate_data.py --count 2000 --categories java_coding,tool_calling

# 指定批量大小（每次 API 调用生成的样本数，默认 5）
python generate_data.py --count 2000 --batch-size 3

# 合并所有数据（种子 + 生成）为训练文件
python generate_data.py --merge

# 数据质量报告
python generate_data.py --report

# 人工审核模式
python generate_data.py --review
```

#### 可插拔样本类型

样本类型定义在 `data/categories/*.yaml` 中，运行时动态加载。**新增类型只需创建 YAML 文件 + 种子数据，不修改任何 Python 代码。**

每个 YAML 文件包含：

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 类型名称 |
| `subcategories` | 是 | 子话题列表 |
| `meta_prompt` | 是 | LLM 生成用 prompt 模板（支持 `{batch_size}` 等占位符） |
| `description` | 否 | 类型描述 |
| `extra_instructions` | 否 | 子话题到详细指令的映射 |

默认三个数据类别及比例：

| 类别 | 比例 | 说明 |
|------|------|------|
| `java_coding` | 50% | Spring Boot/Cloud、MyBatis、WebFlux 等 |
| `tool_calling` | 30% | 工具调用、多步规划、错误恢复 |
| `param_reflection` | 20% | 参数类型/缺失/合理性检查 |

#### 批量生成

默认每次 LLM API 调用生成 5 条样本（通过 `data.batch_size` 或 `--batch-size` 配置）。相比逐条生成，API 调用次数降至约 1/5。

### 2. 训练 (`train.py`)

- QLoRA 4-bit 量化 + Unsloth 优化
- 分层训练/验证集划分（默认 90:10）
- TensorBoard 实时监控（`make tensorboard` 查看曲线）
- 早停机制（验证损失连续不改善时自动停止）
- 每 epoch 输出训练摘要
- 支持从检查点恢复训练
- 训练后自动生成实验记录

```bash
# 基本训练
python train.py

# 指定超参数
python train.py --training.num_epochs 5 --training.learning_rate 3e-5

# 训练后同时合并模型
python train.py --output.merge_model true
```

默认仅保存 LoRA 适配器（~250MB），不执行合并。需要完整模型时使用 `make merge`。

### 3. 评估 (`test_model.py`)

四个评估维度，每个维度输出 0-100 分及评分细项：

| 维度 | 评分因子 |
|------|---------|
| Java 编程 | 关键词命中 (40%) + 代码块完整性 (30%) + 注解/API 正确性 (30%) |
| 工具调用 | 工具选择 (35%) + 格式合规 (20%) + 参数完整性 (25%) + 参数值合理性 (20%) |
| 任务规划 | 规划存在性 (25%) + 步骤编号 (20%) + 工具使用 (20%) + 步骤覆盖 (15%) + 逻辑依赖 (20%) |
| 参数反思 | 结论正确性 (50%) + 反思过程可见性 (25%) + 行为合理性 (25%) |

```bash
# 测试 LoRA 适配器
python test_model.py --model lora_adapter

# 对比模式：原始模型 vs 微调模型
python test_model.py --compare

# 测试原始基础模型
python test_model.py --model /path/to/base/model
```

对比模式输出 `test_results_compare.json`（结构化数据）和 `test_report.md`（可读报告）。

### 4. 实验管理 (`manage_experiments.py`)

每次训练自动生成实验记录到 `experiments/exp-{timestamp}.json`。

```bash
# 列出所有实验
python manage_experiments.py list

# 按验证损失排序
python manage_experiments.py list --sort-by val_loss

# 对比两个实验
python manage_experiments.py compare exp-20260419_130204 exp-20260419_150312
```

### 5. 多 LoRA 合并 (`merge_lora.py`)

将多个独立训练的 LoRA 适配器合并为一个，用于组合不同能力模块。

| 合并方法 | 说明 |
|---------|------|
| `linear` | 加权线性平均（最简单稳定） |
| `ties` | TIES-Merging（稀疏化后合并，保留显著参数） |
| `dare_ties` | DARE + TIES（随机丢弃 + TIES） |
| `cat` | 拼接合并（rank = 所有适配器 rank 之和） |

```bash
# 在 config/config.yaml 的 lora_merge 段配置适配器路径和权重后
python merge_lora.py

# CLI 覆盖
python merge_lora.py --lora_merge.method ties --lora_merge.density 0.3
```

## 模型兼容性

本流水线支持任意兼容 Unsloth 的模型，包括但不限于：

- **Qwen** 系列（Qwen2.5、Qwen3 等）
- **LLaMA** 系列（LLaMA 3、LLaMA 3.1 等）
- **Mistral** 系列

> 始终使用 `tokenizer.apply_chat_template()` 格式化对话，不要手动拼接模板。
> 使用前请在 `config/config.yaml` 的 `model.path` 中设置你的基础模型路径。

## 数据格式

训练数据（`training_data.jsonl`）为 JSONL 格式，每行一个样本。合并后保留 `metadata.category` 字段，供 `train.py` 分层采样使用：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {"category": "java_coding"}}
```

种子样本包含完整 metadata：

```json
{"messages": [...], "metadata": {"category": "java_coding", "subcategory": "springboot_basics", "difficulty": "intermediate"}}
```

`train.py` 优先从 `metadata.category` 读取类别信息进行分层采样；对于无 metadata 的旧数据文件，回退到基于内容的推断。

## 注意事项

- 首次运行 Unsloth 时 Triton kernel 编译较慢，编译缓存保存在 `unsloth_compiled_cache/`
- 部分模型的 4-bit 量化可能有精度损失警告，这是在 VRAM 限制下的权衡
- `config/config.yaml` 已在 `.gitignore` 中，请基于 `config/config.example.yaml` 创建
- `model.path` 必须在训练前设置，代码中没有硬编码的模型路径
- LLM API Key 建议通过环境变量 `LLM_API_KEY` 设置，避免写入配置文件
