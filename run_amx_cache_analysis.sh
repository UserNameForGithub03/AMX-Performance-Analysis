#!/bin/bash

# 专门针对AMX和缓存分析的VTune脚本
# 重点关注AMX利用率、MFU和缓存命中率

set -e

# 配置参数
MODEL_PATH="/data/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
PYTHON_SCRIPT="/data/wangjiaqi/AMX/inference profiling/model_inference_benchmark.py"
OUTPUT_DIR="/data/wangjiaqi/amx_cache_analysis_$(date +%Y%m%d_%H%M%S)"
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

# 运行AMX和缓存分析
run_amx_cache_analysis() {
    echo "=========================================="
    echo "AMX and Cache Analysis for Model Inference"
    echo "=========================================="
    echo "Model: $MODEL_PATH"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="
    
    mkdir -p "$OUTPUT_DIR"
    
    # 使用uarch-exploration收集器，专门分析微架构性能
    echo "Starting AMX and Cache Analysis..."
    $VTUNE_BIN -collect uarch-exploration \
        -knob collect-memory-bandwidth=true \
        -knob collect-frontend-bound=true \
        -knob collect-bad-speculation=true \
        -knob collect-memory-bound=true \
        -knob collect-core-bound=true \
        -knob collect-retiring=true \
        -knob dram-bandwidth-limits=true \
        -result-dir "$OUTPUT_DIR/amx_cache" \
        -data-limit=0 \
        -- python3 "$PYTHON_SCRIPT" --model_path "$MODEL_PATH" --device cpu --benchmark_runs 15 --warmup_runs 5
    
    echo "Analysis completed!"
}

# 生成详细报告
generate_reports() {
    echo "Generating detailed reports..."
    
    # 生成摘要报告
    $VTUNE_BIN -report summary \
        -result-dir "$OUTPUT_DIR/amx_cache" \
        -format text \
        -report-output "$OUTPUT_DIR/amx_cache_summary.txt"
    
    # 生成Top-down分析报告
    $VTUNE_BIN -report top-down \
        -result-dir "$OUTPUT_DIR/amx_cache" \
        -format text \
        -report-output "$OUTPUT_DIR/amx_cache_topdown.txt"
    
    # 生成Bottom-up分析报告
    $VTUNE_BIN -report bottom-up \
        -result-dir "$OUTPUT_DIR/amx_cache" \
        -format text \
        -report-output "$OUTPUT_DIR/amx_cache_bottomup.txt"
    
    # 生成内存访问报告
    $VTUNE_BIN -report memory-access \
        -result-dir "$OUTPUT_DIR/amx_cache" \
        -format text \
        -report-output "$OUTPUT_DIR/amx_cache_memory.txt"
    
    echo "Reports generated successfully!"
}

# 提取关键指标
extract_key_metrics() {
    echo "=========================================="
    echo "Key Performance Metrics"
    echo "=========================================="
    
    if [ -f "$OUTPUT_DIR/amx_cache_summary.txt" ]; then
        echo "=== CPU Utilization and Performance ==="
        grep -E "(CPU Utilization|Elapsed Time|CPU Time|Clockticks|Instructions Retired)" "$OUTPUT_DIR/amx_cache_summary.txt" || echo "No CPU metrics found"
        echo ""
        
        echo "=== Cache Performance ==="
        grep -E "(L1|L2|L3|Cache|Miss|Hit)" "$OUTPUT_DIR/amx_cache_summary.txt" || echo "No cache metrics found"
        echo ""
        
        echo "=== Memory Bandwidth ==="
        grep -E "(Memory|Bandwidth|DRAM|DDR)" "$OUTPUT_DIR/amx_cache_summary.txt" || echo "No memory bandwidth metrics found"
        echo ""
        
        echo "=== AMX and Vector Instructions ==="
        grep -E "(AMX|AVX|Vector|SIMD)" "$OUTPUT_DIR/amx_cache_summary.txt" || echo "No AMX/Vector metrics found"
        echo ""
        
        echo "=== Pipeline Efficiency ==="
        grep -E "(IPC|CPI|Retiring|Frontend|Backend|Bad Speculation)" "$OUTPUT_DIR/amx_cache_summary.txt" || echo "No pipeline metrics found"
        echo ""
    fi
    
    echo "=========================================="
    echo "Detailed analysis files:"
    echo "- Summary: $OUTPUT_DIR/amx_cache_summary.txt"
    echo "- Top-down: $OUTPUT_DIR/amx_cache_topdown.txt"
    echo "- Bottom-up: $OUTPUT_DIR/amx_cache_bottomup.txt"
    echo "- Memory: $OUTPUT_DIR/amx_cache_memory.txt"
    echo "=========================================="
}

# 主函数
main() {
    check_dependencies
    run_amx_cache_analysis
    generate_reports
    extract_key_metrics
    
    echo ""
    echo "Analysis complete! Use the following command to view results:"
    echo "vtune -report summary -result-dir $OUTPUT_DIR/amx_cache"
}

# 运行主函数
main "$@"
