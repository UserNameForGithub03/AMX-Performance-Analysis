#!/bin/bash

# VTune性能分析脚本
# 用于分析大模型推理的AMX利用率、MFU和缓存命中率

set -e

# 配置参数
MODEL_PATH="/data/share/huggingface/hub/models--meta-llama--Llama-3.2-1B"
PYTHON_SCRIPT="/data/wangjiaqi/AMX/model_inference_benchmark.py"
OUTPUT_DIR="/data/wangjiaqi/vtune_analysis_$(date +%Y%m%d_%H%M%S)"
VTUNE_BIN="/opt/intel/oneapi/vtune/latest/bin64/vtune"

# 检查VTune是否安装
if [ ! -f "$VTUNE_BIN" ]; then
    echo "Error: VTune not found at $VTUNE_BIN"
    echo "Please install Intel VTune Profiler or update the path"
    exit 1
fi

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Available models:"
    ls -la /data/share/huggingface/hub/ | grep "models--"
    exit 1
fi

# 检查Python脚本
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "VTune Performance Analysis for Model Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Script: $PYTHON_SCRIPT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 1. 热点分析 (Hotspots Analysis)
echo "Starting Hotspots Analysis..."
$VTUNE_BIN -collect hotspots \
    -result-dir "$OUTPUT_DIR/hotspots" \
    -data-limit=0 \
    -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 5

# 2. 微架构分析 (Microarchitecture Analysis)
echo "Starting Microarchitecture Analysis..."
$VTUNE_BIN -collect uarch-exploration \
    -result-dir "$OUTPUT_DIR/microarch" \
    -data-limit=0 \
    -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 5

# 3. 内存访问分析 (Memory Access Analysis)
echo "Starting Memory Access Analysis..."
$VTUNE_BIN -collect memory-access \
    -result-dir "$OUTPUT_DIR/memory" \
    -data-limit=0 \
    -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 5

# 4. 高级热点分析 (Advanced Hotspots)
echo "Starting Advanced Hotspots Analysis..."
$VTUNE_BIN -collect advanced-hotspots \
    -result-dir "$OUTPUT_DIR/advanced_hotspots" \
    -data-limit=0 \
    -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 5

# 5. 线程分析 (Threading Analysis)
echo "Starting Threading Analysis..."
$VTUNE_BIN -collect threading \
    -result-dir "$OUTPUT_DIR/threading" \
    -data-limit=0 \
    -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 5

# 6. 自定义分析 - 重点关注AMX和缓存
echo "Starting Custom Analysis for AMX and Cache..."
$VTUNE_BIN -collect uarch-exploration \
    -knob enable-tsx=true \
    -knob enable-tsx-abort=true \
    -knob enable-tsx-retry=true \
    -result-dir "$OUTPUT_DIR/amx_cache" \
    -data-limit=0 \
    -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 10

echo "=========================================="
echo "Analysis completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# 生成报告
echo "Generating reports..."

# 生成热点分析报告
if [ -d "$OUTPUT_DIR/hotspots" ]; then
    echo "Generating Hotspots report..."
    $VTUNE_BIN -report summary -result-dir "$OUTPUT_DIR/hotspots" -format text -report-output "$OUTPUT_DIR/hotspots_summary.txt"
    $VTUNE_BIN -report top-down -result-dir "$OUTPUT_DIR/hotspots" -format text -report-output "$OUTPUT_DIR/hotspots_topdown.txt"
fi

# 生成微架构分析报告
if [ -d "$OUTPUT_DIR/microarch" ]; then
    echo "Generating Microarchitecture report..."
    $VTUNE_BIN -report summary -result-dir "$OUTPUT_DIR/microarch" -format text -report-output "$OUTPUT_DIR/microarch_summary.txt"
    $VTUNE_BIN -report top-down -result-dir "$OUTPUT_DIR/microarch" -format text -report-output "$OUTPUT_DIR/microarch_topdown.txt"
fi

# 生成内存分析报告
if [ -d "$OUTPUT_DIR/memory" ]; then
    echo "Generating Memory Access report..."
    $VTUNE_BIN -report summary -result-dir "$OUTPUT_DIR/memory" -format text -report-output "$OUTPUT_DIR/memory_summary.txt"
    $VTUNE_BIN -report top-down -result-dir "$OUTPUT_DIR/memory" -format text -report-output "$OUTPUT_DIR/memory_topdown.txt"
fi

# 生成AMX和缓存分析报告
if [ -d "$OUTPUT_DIR/amx_cache" ]; then
    echo "Generating AMX and Cache report..."
    $VTUNE_BIN -report summary -result-dir "$OUTPUT_DIR/amx_cache" -format text -report-output "$OUTPUT_DIR/amx_cache_summary.txt"
    $VTUNE_BIN -report top-down -result-dir "$OUTPUT_DIR/amx_cache" -format text -report-output "$OUTPUT_DIR/amx_cache_topdown.txt"
fi

echo "=========================================="
echo "All reports generated!"
echo "Check the following files for detailed analysis:"
echo "- $OUTPUT_DIR/hotspots_summary.txt (CPU hotspots)"
echo "- $OUTPUT_DIR/microarch_summary.txt (Microarchitecture metrics)"
echo "- $OUTPUT_DIR/memory_summary.txt (Memory access patterns)"
echo "- $OUTPUT_DIR/amx_cache_summary.txt (AMX and cache utilization)"
echo "=========================================="

# 显示关键指标摘要
echo "Key Performance Metrics Summary:"
echo "================================"

if [ -f "$OUTPUT_DIR/amx_cache_summary.txt" ]; then
    echo "AMX and Cache Metrics:"
    grep -E "(AMX|Cache|L1|L2|L3|Memory|Bandwidth)" "$OUTPUT_DIR/amx_cache_summary.txt" | head -20
    echo ""
fi

if [ -f "$OUTPUT_DIR/microarch_summary.txt" ]; then
    echo "Microarchitecture Metrics:"
    grep -E "(IPC|CPI|Retiring|Frontend|Backend|Bad Speculation)" "$OUTPUT_DIR/microarch_summary.txt" | head -20
    echo ""
fi

echo "Analysis complete! Use 'vtune -report' command to view detailed results."
echo "Example: vtune -report summary -result-dir $OUTPUT_DIR/amx_cache"
