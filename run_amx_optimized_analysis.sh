#!/bin/bash

# AMX优化版本的VTune性能分析脚本 (Qwen3-32B)
# 使用IPEX和BF16精度来触发AMX指令

set -e

# 激活conda环境
source /data/wangjiaqi/anaconda3/etc/profile.d/conda.sh
conda activate openvivo

# 配置参数
MODEL_PATH="/data/share/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
# MODEL_PATH="/data/share/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PYTHON_SCRIPT="/data/wangjiaqi/AMX/model_inference_amx_optimized.py"
OUTPUT_DIR="/data/wangjiaqi/amx_optimized_analysis_qwen3-32b_$(date +%Y%m%d_%H%M%S)"
VTUNE_BIN="/opt/intel/oneapi/vtune/latest/bin64/vtune"

# 检查依赖
check_dependencies() {
    if [ ! -f "$VTUNE_BIN" ]; then
        echo "Error: VTune not found at $VTUNE_BIN"
        echo "Please install Intel VTune Profiler"
        exit 1
    fi
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model path not found: $MODEL_PATH"
        exit 1
    fi
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "Error: Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# 运行AMX优化分析
run_amx_optimized_analysis() {
    echo "=========================================="
    echo "AMX-Optimized Model Inference Analysis (Qwen3-32B)"
    echo "=========================================="
    echo "Model: $MODEL_PATH"
    echo "Output: $OUTPUT_DIR"
    echo "Script: $PYTHON_SCRIPT"
    echo "=========================================="
    
    mkdir -p "$OUTPUT_DIR"
    
    # 使用uarch-exploration收集器，专门分析微架构性能
    echo "Starting AMX-Optimized Analysis..."
    $VTUNE_BIN -collect uarch-exploration \
        -knob collect-memory-bandwidth=true \
        -knob collect-frontend-bound=true \
        -knob collect-bad-speculation=true \
        -knob collect-memory-bound=true \
        -knob collect-core-bound=true \
        -knob collect-retiring=true \
        -knob dram-bandwidth-limits=true \
        -result-dir "$OUTPUT_DIR/amx_optimized" \
        -data-limit=0 \
        -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 10 --warmup_runs 3 --use_bf16
    
    echo "Analysis completed!"
}

# 生成详细报告
generate_reports() {
    echo "Generating detailed reports..."
    
    # 生成摘要报告
    echo "Generating summary report..."
    $VTUNE_BIN -report summary \
        -result-dir "$OUTPUT_DIR/amx_optimized" \
        -format text \
        -report-output "$OUTPUT_DIR/amx_optimized_summary.txt" > /dev/null 2>&1
    
    # 生成Top-down分析报告
    echo "Generating top-down report..."
    $VTUNE_BIN -report top-down \
        -result-dir "$OUTPUT_DIR/amx_optimized" \
        -format text \
        -report-output "$OUTPUT_DIR/amx_optimized_topdown.txt" > /dev/null 2>&1
    
    # 验证报告文件是否创建成功
    echo "Verifying report files..."
    if [ -f "$OUTPUT_DIR/amx_optimized_summary.txt" ]; then
        echo "✓ Summary report created: $(wc -l < "$OUTPUT_DIR/amx_optimized_summary.txt") lines"
    else
        echo "✗ Summary report not found"
    fi
    
    if [ -f "$OUTPUT_DIR/amx_optimized_topdown.txt" ]; then
        echo "✓ Top-down report created: $(wc -l < "$OUTPUT_DIR/amx_optimized_topdown.txt") lines"
    else
        echo "✗ Top-down report not found"
    fi
    
    echo "Reports generated successfully!"
}

# 提取关键指标
extract_key_metrics() {
    echo "=========================================="
    echo "Key Performance Metrics (AMX-Optimized)"
    echo "=========================================="
    
    if [ -f "$OUTPUT_DIR/amx_optimized_summary.txt" ]; then
        echo "=== CPU Utilization and Performance ==="
        grep -E "(CPU Utilization|Elapsed Time|CPU Time|Clockticks|Instructions Retired)" "$OUTPUT_DIR/amx_optimized_summary.txt" || echo "No CPU metrics found"
        echo ""
        
        echo "=== Cache Performance ==="
        grep -E "(L1|L2|L3|Cache|Miss|Hit)" "$OUTPUT_DIR/amx_optimized_summary.txt" || echo "No cache metrics found"
        echo ""
        
        echo "=== Memory Bandwidth ==="
        grep -E "(Memory|Bandwidth|DRAM|DDR)" "$OUTPUT_DIR/amx_optimized_summary.txt" || echo "No memory bandwidth metrics found"
        echo ""
        
        echo "=== AMX and Vector Instructions ==="
        grep -E "(AMX|AVX|Vector|SIMD)" "$OUTPUT_DIR/amx_optimized_summary.txt" || echo "No AMX/Vector metrics found"
        echo ""
        
        echo "=== Pipeline Efficiency ==="
        grep -E "(IPC|CPI|Retiring|Frontend|Backend|Bad Speculation)" "$OUTPUT_DIR/amx_optimized_summary.txt" || echo "No pipeline metrics found"
        echo ""
        
        echo "=== AMX Utilization (Key Metric) ==="
        grep -E "AMX Busy" "$OUTPUT_DIR/amx_optimized_summary.txt" || echo "AMX Busy metric not found"
        echo ""
    fi
    
    echo "=========================================="
    echo "Detailed analysis files:"
    echo "- Summary: $OUTPUT_DIR/amx_optimized_summary.txt"
    echo "- Top-down: $OUTPUT_DIR/amx_optimized_topdown.txt"
    echo "=========================================="
}

# 主函数
main() {
    check_dependencies
    run_amx_optimized_analysis
    generate_reports
    # OUTPUT_DIR="/data/wangjiaqi/amx_optimized_analysis_qwen3-32b_20251029_154917"
    extract_key_metrics
    
    echo ""
    echo "Analysis complete! Use the following command to view results:"
    echo "vtune -report summary -result-dir $OUTPUT_DIR/amx_optimized"
}

# 运行主函数
main "$@"
