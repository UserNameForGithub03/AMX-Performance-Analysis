#!/bin/bash
# 一键启动双模型AMX分析
# 直接运行: bash /data/wangjiaqi/AMX/RUN_THIS.sh

cd /data/wangjiaqi/AMX

# 设置权限
chmod +x run_amx_optimized_analysis_8b.sh
chmod +x run_amx_optimized_analysis.sh

# 创建日志文件名
LOG_8B="/data/wangjiaqi/vtune_8b_log_$(date +%Y%m%d_%H%M%S).txt"
LOG_32B="/data/wangjiaqi/vtune_32b_log_$(date +%Y%m%d_%H%M%S).txt"

# 启动Qwen3-8B分析
echo "Starting Qwen3-8B analysis in tmux session 'vtune-8b'..."
tmux new-session -d -s vtune-8b
tmux send-keys -t vtune-8b "cd /data/wangjiaqi/AMX && bash run_amx_optimized_analysis_8b.sh 2>&1 | tee $LOG_8B" C-m

# 启动Qwen3-32B分析
echo "Starting Qwen3-32B analysis in tmux session 'vtune-32b'..."
tmux new-session -d -s vtune-32b
tmux send-keys -t vtune-32b "cd /data/wangjiaqi/AMX && bash run_amx_optimized_analysis.sh 2>&1 | tee $LOG_32B" C-m

echo ""
echo "=========================================="
echo "✓ Both analyses started successfully!"
echo "=========================================="
echo ""
echo "📊 Monitor progress:"
echo "  tmux attach -t vtune-8b   # View Qwen3-8B"
echo "  tmux attach -t vtune-32b  # View Qwen3-32B"
echo ""
echo "📝 Log files:"
echo "  $LOG_8B"
echo "  $LOG_32B"
echo ""
echo "🔄 Check status:"
echo "  tmux ls"
echo ""
echo "⌨️  Detach from tmux: Ctrl+B, then D"
echo "=========================================="


