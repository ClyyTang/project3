#!/bin/bash

# CoT生成流程总控脚本
# 自动运行三个步骤：observations -> subtasks -> cot

set -e  # 遇到错误立即退出

WORK_DIR="/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation"
OUTPUT_DIR="${WORK_DIR}/outputs"

echo "======================================"
echo "CoT生成流程"
echo "======================================"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 步骤1: 生成observations
echo "[步骤1/3] 生成当前观察..."
python3 ${WORK_DIR}/location_observer.py
echo "✅ 步骤1完成"
echo ""

# 步骤2: 生成subtasks
echo "[步骤2/3] 生成子任务..."
python3 ${WORK_DIR}/subtask_generation.py
echo "✅ 步骤2完成"
echo ""

# 步骤3: 生成CoT
echo "[步骤3/3] 生成思维链..."
python3 ${WORK_DIR}/cot_dataset_generator.py
echo "✅ 步骤3完成"
echo ""

echo "======================================"
echo "✅ 所有步骤完成！"
echo "======================================"
echo ""
echo "输出文件："
echo "  - ${OUTPUT_DIR}/observations.json"
echo "  - ${OUTPUT_DIR}/subtasks.json"
echo "  - ${OUTPUT_DIR}/cot_dataset.json"