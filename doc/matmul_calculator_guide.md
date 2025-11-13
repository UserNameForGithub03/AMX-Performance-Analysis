# 矩阵乘法计算工具使用指南

本指南说明 `amx_matmul_calculator.py` 和 `avx_matmul_calculator.py` 的使用方法和原理。

## 工具概述

### amx_matmul_calculator.py
- 用于 AMX（Advanced Matrix Extensions）矩阵乘法性能计算
- 支持数据类型：BF16、INT8、FP32

### avx_matmul_calculator.py
- 用于 AVX（Advanced Vector Extensions）矩阵乘法性能计算
- 支持 AVX512 和 AVX256（AVX2）
- 支持数据类型：BF16、INT8、FP32

## 基本原理

### 1. 操作数计算

矩阵乘法 **C = A × B**（A: M×K, B: K×N, C: M×N）：
- 总操作数：`M × N × K` 次乘加操作
- 总 FLOPS：`2 × M × N × K`（乘加 = 2 FLOPS）

### 2. 每周期操作数（理论峰值）

#### AMX
- BF16: **1024 ops/cycle**
- INT8: **2048 ops/cycle**
- FP32: **1024 ops/cycle**

#### AVX
- **AVX512 BF16**: **512 ops/cycle**（2个FMA单元 × 256 ops/FMA）
- **AVX256 BF16**: **256 ops/cycle**（2个FMA单元 × 128 ops/FMA）

### 3. 理论时间计算

```python
理论时间 = 总操作数 / (每周期操作数 × CPU频率)
```

**重要说明**：
- 理论时间基于 **单核心、单AMX/AVX单元** 计算
- 假设所有数据都在缓存中
- 没有考虑内存访问延迟、流水线停顿等实际开销

### 4. CPU 核心资源

**每个 CPU 核心包含**：
- 1 个 AMX 单元
- 2 个 AVX-512 FMA 单元

因此理论计算假设单核心使用，实际 benchdnn 默认使用所有核心（如192个）。

## 使用方法

### AMX 计算器

```bash
# 基本使用（自动检测CPU频率）
python amx_matmul_calculator.py --M 1024 --K 1024 --N 1024

# 指定数据类型
python amx_matmul_calculator.py --M 1024 --K 1024 --N 1024 --dtype bf16

# 指定CPU频率
python amx_matmul_calculator.py --M 1024 --K 1024 --N 1024 --freq 3.44

# 跳过实际性能测试
python amx_matmul_calculator.py --M 1024 --K 1024 --N 1024 --no-benchmark
```

### AVX 计算器

```bash
# 使用 AVX512
python avx_matmul_calculator.py --M 1024 --K 1024 --N 1024 --avx avx512

# 使用 AVX256 (AVX2)
python avx_matmul_calculator.py --M 1024 --K 1024 --N 1024 --avx avx256

# 指定数据类型
python avx_matmul_calculator.py --M 1024 --K 1024 --N 1024 \
    --avx avx512 --dtype fp32
```

## 重要注意事项

### 1. 理论时间 vs 实际时间

**理论时间**：
- 基于单核心、单AMX/AVX单元
- 假设完美的缓存命中
- 不考虑实际开销

**实际时间（benchdnn）**：
- 默认使用所有可用核心（192个）
- 包含 cold cache 机制（模拟缓存未命中）
- 包含系统开销

因此：
- **理论时间（单核心）vs 实际时间（多核心）** 的对比不公平
- 建议设置 `OMP_NUM_THREADS=1` 来公平对比

### 2. 小矩阵的固定开销

对于极小的矩阵（如 8×256:256×8）：
- 固定开销（kernel启动、内存管理等）远大于计算时间
- benchdnn 的时间结果主要反映固定开销，而非计算性能
- 建议对极小矩阵使用批量测试

### 3. 矩阵维度对齐

**AMX 性能优化**：
- 512×512 输出矩阵通常有专门的高度优化实现
- 建议尽量使用 512 的倍数作为输出维度
- 1024×512 可能比 512×512 慢很多（即使操作数只差2倍）

### 4. 线程配置

benchdnn 默认使用所有可用核心。要测试单核心性能：
```bash
OMP_NUM_THREADS=1 python amx_matmul_calculator.py ...
```

## 输出说明

工具会输出：

1. **操作数计算**：总操作数、FLOPS
2. **理论性能参数**：每周期操作数
3. **理论计算时间**：基于单核心的理论时间
4. **实际性能测试**（如果启用）：
   - 实际耗时
   - 效率（理论时间/实际时间）
   - 实际吞吐量

## 计算公式总结

```
总操作数 = M × N × K
理论cycles = 总操作数 / 每周期操作数
理论时间 = 理论cycles / (CPU频率 × 10⁹)
理论吞吐量 = 总FLOPS / 理论时间 / 10⁹ (GFLOPS)
```

## 代码位置

- `amx_matmul_calculator.py` - AMX计算器主程序
- `avx_matmul_calculator.py` - AVX计算器主程序
- 关键函数：
  - `calculate_matmul_ops()` - 计算操作数
  - `calculate_theoretical_time()` - 计算理论时间
  - `run_benchdnn()` - 运行实际性能测试

