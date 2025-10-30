

```
.
├── measure_amx_util.py  #perf测matmul利用率，python measure_amx_util.py matmul --m 1024 --k 1024 --n 1024 --dtype bf16
├── model_inference_amx_optimized.py #启用AMX的LLM Inference
├── model_inference_benchmark.py #没有特定优化的LLM Inference
├── quick_test.py  #快速验证模型加载和基本推理（路径，正常加载，简单文本生成） python quick_test.py
├── run_amx_cache_analysis.sh  #AMX利用率和缓存命中率qw3-0.6B
├── run_amx_optimized_analysis.sh  #配合model_inference_amx_optimized.py
└── run_vtune_analysis.sh   #综合Vtune性能分析
```
### 流程
1. quick_test.py 
   ↓ (验证模型可用)
   
2. model_inference_benchmark.py 
   ↓ (获取基准性能)
   
3. model_inference_amx_optimized.py 
   ↓ (测试AMX优化效果)
   
4. measure_amx_util.py 
   ↓ (量化AMX利用率)
   
5. run_amx_cache_analysis.sh 或 run_vtune_analysis.sh
   ↓ (深入性能分析)
   
6. 根据VTune报告优化代码