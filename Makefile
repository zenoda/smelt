# ============================================================
# Smelt - QLoRA Training Pipeline Makefile
# ============================================================
#
# 用法:
#   make data         生成训练数据（含合并）
#   make train        运行 QLoRA 训练
#   make test         运行模型评估
#   make all          完整流程: data -> train -> test
#   make clean        清理生成的文件（带确认提示）
#   make tensorboard  启动 TensorBoard
#   make check        检查环境依赖
#   make merge        合并 LoRA 适配器到基础模型
#   make merge-lora   合并多个 LoRA 适配器
#
# 所有目标自动激活 venv 环境
# ============================================================

SHELL := /bin/bash

# 包版本检查脚本
define PACKAGE_CHECK_SCRIPT
import importlib
pkgs = [("torch","torch"), ("transformers","transformers"), ("peft","peft"), ("trl","trl"), ("unsloth","unsloth"), ("datasets","datasets"), ("accelerate","accelerate"), ("tensorboard","tensorboard"), ("pyyaml","yaml")]
for display, mod in pkgs:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "installed")
        print(f"  {display:20s} {v}")
    except ImportError:
        print(f"  {display:20s} [未安装]")
endef
export PACKAGE_CHECK_SCRIPT

# 虚拟环境路径
VENV := venv
ACTIVATE := source $(VENV)/bin/activate

# Python 命令（通过 venv 执行）
PYTHON := $(ACTIVATE) && python

# 关键路径（与 config/config.example.yaml 默认值一致）
TRAINING_DATA := training_data.jsonl
ADAPTER_DIR := lora_adapter
MERGED_DIR := merged_model
SEEDS_DIR := data/seeds
GENERATED_DIR := data/generated
RUNS_DIR := runs
EXPERIMENTS_DIR := experiments

# ============================================================
# 帮助信息（默认目标）
# ============================================================
.PHONY: help
help:
	@echo "============================================================"
	@echo "  Smelt - QLoRA Training Pipeline"
	@echo "============================================================"
	@echo ""
	@echo "可用目标:"
	@echo "  make data         生成训练数据（含合并）"
	@echo "  make train        运行 QLoRA 训练"
	@echo "  make test         运行模型评估"
	@echo "  make all          完整流程: data -> train -> test"
	@echo "  make clean        清理生成的文件（带确认提示）"
	@echo "  make tensorboard  启动 TensorBoard"
	@echo "  make check        检查环境依赖"
	@echo "  make merge        合并 LoRA 适配器到基础模型"
	@echo "  make merge-lora   合并多个 LoRA 适配器"
	@echo ""
	@echo "环境要求:"
	@echo "  - Python venv 位于 ./venv/"
	@echo "  - config/config.yaml（或 config/config.example.yaml 回退）"
	@echo "  - GPU: Nvidia RTX 2080 Ti (22GB VRAM)"
	@echo ""

# ============================================================
# 前置检查函数
# ============================================================

# 检查 venv 是否存在
define check_venv
	@if [ ! -d "$(VENV)" ]; then \
		echo "[错误] 虚拟环境不存在: $(VENV)/"; \
		echo "请先创建: python -m venv $(VENV) && source $(VENV)/bin/activate && pip install -r requirements.txt"; \
		exit 1; \
	fi
endef

# 检查训练数据是否存在
define check_training_data
	@if [ ! -f "$(TRAINING_DATA)" ]; then \
		echo "[错误] 训练数据文件不存在: $(TRAINING_DATA)"; \
		echo "请先运行: make data"; \
		exit 1; \
	fi
endef

# 检查适配器是否存在
define check_adapter
	@if [ ! -d "$(ADAPTER_DIR)" ]; then \
		echo "[错误] LoRA 适配器不存在: $(ADAPTER_DIR)/"; \
		echo "请先运行: make train"; \
		exit 1; \
	fi
endef

# 检查种子文件是否存在
define check_seeds
	@if [ ! -d "$(SEEDS_DIR)" ] || [ -z "$$(ls -A $(SEEDS_DIR)/*.jsonl 2>/dev/null)" ]; then \
		echo "[错误] 种子样本目录为空或不存在: $(SEEDS_DIR)/"; \
		echo "请确保 data/seeds/ 下有 .jsonl 种子文件。"; \
		exit 1; \
	fi
endef

# 检查配置文件
define check_config
	@if [ ! -f "config/config.yaml" ] && [ ! -f "config/config.example.yaml" ]; then \
		echo "[错误] 找不到配置文件（config/config.yaml 或 config/config.example.yaml）"; \
		echo "请先创建: cp config/config.example.yaml config/config.yaml"; \
		exit 1; \
	fi
endef

# ============================================================
# 数据生成
# ============================================================
.PHONY: data
data:
	$(call check_venv)
	$(call check_seeds)
	$(call check_config)
	@echo "============================================================"
	@echo "  数据生成 + 合并"
	@echo "============================================================"
	$(PYTHON) generate_data.py
	@echo ""
	@echo "正在合并数据..."
	$(PYTHON) generate_data.py --merge
	@echo ""
	@echo "[完成] 训练数据已生成: $(TRAINING_DATA)"
	@if [ -f "$(TRAINING_DATA)" ]; then \
		lines=$$(wc -l < "$(TRAINING_DATA)"); \
		echo "  样本数: $$lines"; \
	fi

# ============================================================
# 训练
# ============================================================
.PHONY: train
train:
	$(call check_venv)
	$(call check_training_data)
	$(call check_config)
	@echo "============================================================"
	@echo "  QLoRA 训练"
	@echo "============================================================"
	$(PYTHON) train.py
	@echo ""
	@echo "[完成] 训练结束"
	@if [ -d "$(ADAPTER_DIR)" ]; then \
		echo "  适配器路径: $(ADAPTER_DIR)/"; \
	fi

# ============================================================
# 评估测试
# ============================================================
.PHONY: test
test:
	$(call check_venv)
	$(call check_config)
	@echo "============================================================"
	@echo "  模型评估"
	@echo "============================================================"
	@if [ -d "$(ADAPTER_DIR)" ]; then \
		echo "  检测到适配器: $(ADAPTER_DIR)/"; \
		echo "  运行对比评估（原始模型 vs 微调模型）..."; \
		$(ACTIVATE) && python test_model.py --compare; \
	else \
		echo "  未检测到适配器，运行基础模型评估..."; \
		$(ACTIVATE) && python test_model.py; \
	fi
	@echo ""
	@echo "[完成] 评估结束"

# ============================================================
# 完整流程
# ============================================================
.PHONY: all
all: data train test
	@echo ""
	@echo "============================================================"
	@echo "  完整流程已完成: data -> train -> test"
	@echo "============================================================"

# ============================================================
# 清理
# ============================================================
.PHONY: clean
clean:
	@echo "以下文件/目录将被删除:"
	@echo ""
	@[ -f "$(TRAINING_DATA)" ] && echo "  $(TRAINING_DATA)" || true
	@[ -d "$(GENERATED_DIR)" ] && echo "  $(GENERATED_DIR)/" || true
	@[ -d "$(ADAPTER_DIR)" ] && echo "  $(ADAPTER_DIR)/" || true
	@[ -d "$(MERGED_DIR)" ] && echo "  $(MERGED_DIR)/" || true
	@[ -d "$(RUNS_DIR)" ] && echo "  $(RUNS_DIR)/" || true
	@[ -d "$(EXPERIMENTS_DIR)" ] && echo "  $(EXPERIMENTS_DIR)/" || true
	@for dir in outputs_*/; do [ -d "$$dir" ] && echo "  $$dir" || true; done
	@[ -f "data_report.json" ] && echo "  data_report.json" || true
	@[ -f "rejected_samples.jsonl" ] && echo "  rejected_samples.jsonl" || true
	@[ -f "test_results.json" ] && echo "  test_results.json" || true
	@[ -f "test_results_compare.json" ] && echo "  test_results_compare.json" || true
	@[ -f "test_report.md" ] && echo "  test_report.md" || true
	@echo ""
	@read -p "确认删除以上文件？[y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo ""; \
		echo "正在清理..."; \
		rm -f "$(TRAINING_DATA)"; \
		rm -rf "$(GENERATED_DIR)"; \
		rm -rf "$(ADAPTER_DIR)"; \
		rm -rf "$(MERGED_DIR)"; \
		rm -rf "$(RUNS_DIR)"; \
		rm -rf "$(EXPERIMENTS_DIR)"; \
		rm -rf outputs_*/; \
		rm -f data_report.json rejected_samples.jsonl; \
		rm -f test_results.json test_results_compare.json test_report.md; \
		echo "[完成] 清理完毕"; \
	else \
		echo "已取消"; \
	fi

# ============================================================
# TensorBoard
# ============================================================
.PHONY: tensorboard
tensorboard:
	$(call check_venv)
	@if [ ! -d "$(RUNS_DIR)" ] || [ -z "$$(ls -A $(RUNS_DIR) 2>/dev/null)" ]; then \
		echo "[警告] TensorBoard 日志目录为空或不存在: $(RUNS_DIR)/"; \
		echo "请先运行训练: make train"; \
		exit 1; \
	fi
	@echo "============================================================"
	@echo "  启动 TensorBoard"
	@echo "============================================================"
	@echo "  日志目录: $(RUNS_DIR)/"
	@echo "  访问地址: http://localhost:6006"
	@echo "  按 Ctrl+C 停止"
	@echo ""
	$(ACTIVATE) && tensorboard --logdir $(RUNS_DIR)

# ============================================================
# 环境检查
# ============================================================
.PHONY: check
check:
	@echo "============================================================"
	@echo "  环境检查报告"
	@echo "============================================================"
	@echo ""
	@echo "-- 系统信息 --"
	@echo "  操作系统: $$(uname -s -r)"
	@echo "  主机名:   $$(hostname)"
	@echo ""
	@echo "-- GPU 信息 --"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		gpu_name=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
		gpu_vram=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1); \
		gpu_used=$$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1); \
		gpu_driver=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); \
		echo "  GPU:       $$gpu_name"; \
		echo "  VRAM:      $$gpu_vram (已用: $$gpu_used)"; \
		echo "  驱动版本:  $$gpu_driver"; \
	else \
		echo "  [错误] nvidia-smi 不可用"; \
	fi
	@echo ""
	@echo "-- Python 环境 --"
	@if [ -d "$(VENV)" ]; then \
		echo "  venv:      $(VENV)/ [存在]"; \
		py_ver=$$($(ACTIVATE) && python --version 2>&1); \
		echo "  Python:    $$py_ver"; \
	else \
		echo "  venv:      $(VENV)/ [不存在]"; \
	fi
	@echo ""
	@echo "-- 关键包版本 --"
	@if [ -d "$(VENV)" ]; then \
		$(ACTIVATE) && python -c "$$PACKAGE_CHECK_SCRIPT" 2>&1 | grep -v "^Skipping"; \
	else \
		echo "  [跳过] venv 不存在"; \
	fi
	@echo ""
	@echo "-- 配置文件 --"
	@if [ -f "config/config.yaml" ]; then \
		echo "  config/config.yaml:         [存在]"; \
	else \
		echo "  config/config.yaml:         [不存在]"; \
	fi
	@if [ -f "config/config.example.yaml" ]; then \
		echo "  config/config.example.yaml: [存在]"; \
	else \
		echo "  config/config.example.yaml: [不存在]"; \
	fi
	@echo ""
	@echo "-- 数据文件 --"
	@if [ -d "$(SEEDS_DIR)" ]; then \
		seed_count=$$(cat $(SEEDS_DIR)/*.jsonl 2>/dev/null | wc -l); \
		echo "  种子样本:   $$seed_count 条 ($(SEEDS_DIR)/)"; \
	else \
		echo "  种子样本:   [目录不存在]"; \
	fi
	@if [ -f "$(TRAINING_DATA)" ]; then \
		data_count=$$(wc -l < "$(TRAINING_DATA)"); \
		echo "  训练数据:   $$data_count 条 ($(TRAINING_DATA))"; \
	else \
		echo "  训练数据:   [不存在]"; \
	fi
	@echo ""
	@echo "-- 模型产物 --"
	@if [ -d "$(ADAPTER_DIR)" ]; then \
		echo "  LoRA 适配器: $(ADAPTER_DIR)/ [存在]"; \
	else \
		echo "  LoRA 适配器: $(ADAPTER_DIR)/ [不存在]"; \
	fi
	@if [ -d "$(MERGED_DIR)" ]; then \
		echo "  合并模型:    $(MERGED_DIR)/ [存在]"; \
	else \
		echo "  合并模型:    $(MERGED_DIR)/ [不存在]"; \
	fi
	@echo ""
	@echo "============================================================"

# ============================================================
# LoRA 适配器合并到基础模型
# ============================================================
.PHONY: merge
merge:
	$(call check_venv)
	$(call check_adapter)
	$(call check_config)
	@echo "============================================================"
	@echo "  合并 LoRA 适配器到基础模型"
	@echo "============================================================"
	@echo "  适配器: $(ADAPTER_DIR)/"
	@echo "  输出:   $(MERGED_DIR)/"
	@echo ""
	$(PYTHON) train.py --output.merge_model true
	@echo ""
	@echo "[完成] 合并模型已保存到: $(MERGED_DIR)/"

# ============================================================
# 多 LoRA 适配器合并
# ============================================================
.PHONY: merge-lora
merge-lora:
	$(call check_venv)
	$(call check_config)
	@echo "============================================================"
	@echo "  多 LoRA 适配器合并"
	@echo "============================================================"
	$(PYTHON) merge_lora.py
	@echo ""
	@echo "[完成] 合并完成"

# ============================================================
# 默认目标
# ============================================================
.DEFAULT_GOAL := help
