

```
目录结构：
|── measure_amx_util.py  # torch做矩阵乘法看exe.amx_busy/cycles
├── model_inference_amx_optimized.py #LLM Inference
├── model_inference_benchmark.py
├── quick_test.py  #快速验证模型加载和基本推理（路径，正常加载，简单文本生成） python quick_test.py
├── results
│   ├── results_32b.txt
│   ├── results_qwen3-32b_summary.txt
├── run_amx_cache_analysis.sh
├── run_amx_optimized_analysis.sh #配合model_inference_amx_optimized.py
└── run_vtune_analysis.sh
```
### 流程
1. quick_test.py 
   (验证模型可用)
   
2. model_inference_amx_optimized.py 
   (测试AMX优化效果)
   
3. run_amx_cache_analysis.sh 或 run_vtune_analysis.sh
   (深入性能分析)

### 一些命令
- MxK和KxN矩阵乘法计算iters次 `./measure_amx_util.py matmul --m 1 --k 7168 --n 28672 --dtype bf16 --iters 50`
- 生成某模型Inference的VTune报告 `run_amx_optimized_analysis.sh`，会生成如下结构的文件夹。
``` /data/wangjiaqi/amx_optimized_analysis_qwen3-8b_20251029_155140
├── amx_optimized
│   ├── amx_optimized.vtune
│   ├── archive
│   ├── config
│   ├── data.0
│   ├── log
│   └── sqlite-db
├── amx_optimized_summary.txt
└── amx_optimized_topdown.txt
```