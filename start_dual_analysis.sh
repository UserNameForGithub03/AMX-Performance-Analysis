#!/bin/bash

# 在tmux中同时运行Qwen3-8B和Qwen3-32B的AMX性能分析
# 使用方法: bash start_dual_analysis.sh

echo "=========================================="
echo "Starting Dual AMX Performance Analysis"
echo "=========================================="
echo "Model 1: Qwen3-8B"
echo "Model 2: Qwen3-32B"
echo "=========================================="

# 设置脚本可执行权限
chmod +x /data/wangjiaqi/AMX/run_amx_optimized_analysis_8b.sh
chmod +x /data/wangjiaqi/AMX/run_amx_optimized_analysis.sh

# 创建tmux会话并运行Qwen3-8B分析
echo "Starting Qwen3-8B analysis in tmux session 'vtune-8b'..."
tmux new-session -d -s vtune-8b "bash /data/wangjiaqi/AMX/run_amx_optimized_analysis_8b.sh 2>&1 | tee /data/wangjiaqi/vtune_8b_log_$(date +%Y%m%d_%H%M%S).txt"

# 等待1秒
sleep 1

# 创建tmux会话并运行Qwen3-32B分析
echo "Starting Qwen3-32B analysis in tmux session 'vtune-32b'..."
tmux new-session -d -s vtune-32b "bash /data/wangjiaqi/AMX/run_amx_optimized_analysis.sh 2>&1 | tee /data/wangjiaqi/vtune_32b_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "=========================================="
echo "Both analyses started successfully!"
echo "=========================================="
echo ""
echo "To monitor progress:"
echo "  - Qwen3-8B:  tmux attach -t vtune-8b"
echo "  - Qwen3-32B: tmux attach -t vtune-32b"
echo ""
echo "To list all tmux sessions:"
echo "  tmux ls"
echo ""
echo "To detach from tmux (without stopping):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "Log files are being saved to:"
echo "  - /data/wangjiaqi/vtune_8b_log_*.txt"
echo "  - /data/wangjiaqi/vtune_32b_log_*.txt"
echo ""
echo "Estimated completion time: 1-3 hours per model"
echo "=========================================="


