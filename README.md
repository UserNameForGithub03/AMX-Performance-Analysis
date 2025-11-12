

```
目录结构：

├── amx_matmul_calculator.py    # 理论/实际
├── avx_matmul_calculator.py
├── frequency
│   ├── bind-and-test.sh
│   ├── fix-core-freq.sh        # 需要sudo权限 固定CPU频率在3.9~4.0GHz ./fix-core-freq.sh <core_id>
│   └── unfix-core-freq.sh      # 需要sudo权限
|── measure_amx_util.py         # torch做矩阵乘法看exe.amx_busy/cycles
├── model_inference_amx_optimized.py #LLM Inference全过程
├── model_inference_benchmark.py
├── quick_test.py  #快速验证模型加载和基本推理（路径，正常加载，简单文本生成） python quick_test.py
├── results                     # from vtune
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
- 用benchdnn中的matmul driver的示例命令（绑定core8， 算10000次矩阵乘法）：   
`DNNL_MAX_CPU_ISA=AVX512_CORE_AMX taskset -c 8 /data/wangjiaqi/oneDNN/build/tests/benchdnn/benchdnn --matmul --mode=P --fix-times-per-prb=10000 --dt=bf16:bf16:bf16 --stag=any --wtag=any --dtag=any 512x512:512x512`    
`DNNL_MAX_CPU_ISA=AVX512_CORE_BF16`
- 绑定某个core`taskset -c <core_id>`, 查看某个core的频率`watch -n <freq> "awk '/^processor\t+:\s+<core_id>$/,/^$/ {if(/cpu MHz/) print}' /proc/cpuinfo"`, 查看前10个core的频率`watch -n <freq> "cat /proc/cpuinfo | grep -i mhz | head"`。   
例如`watch -n 0.3 "awk '/^processor\t+:\s+8$/,/^$/ {if(/cpu MHz/) print}' /proc/cpuinfo"`, 每隔0.3s查看一次core 8 的频率
- 对于amx_matmul_calculator.py和avx_matmul_calculator，理论值由op数 op/cycle和CPU频率共同估算，是单个core的情况。 使用方式：   
`python3 ./amx_matmul_calculator.py --M 1024 --N 1024 --K 1024 --dtype bf16 --freq 3.5`   
`python3 ./amx_matmul_calculator.py` 也行。