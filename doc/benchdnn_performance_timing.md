# benchdnn 性能计时机制详解

本文档说明 benchdnn 中 `avg(ms)`、`min(ms)` 等性能指标的计算原理和代码位置。

## 概述

benchdnn 的 matmul 测试中，**`avg(ms)` 只测量 primitive 执行本身的时间**，不包括内存分配、数据填充、初始化等开销。

## 执行流程与计时起点/终点

### 主要执行路径

```
matmul.cpp::doit() [入口]
  ↓
  1. init_memory_args()          [不计时]
  2. init_ref_memory_args()      [不计时]
  3. execute_and_wait()          [不计时，用于验证/预热]
  4. check_correctness()         [不计时]
  5. measure_perf()              [← 从这里开始计时]
     ↓
     measure_perf_individual()   [CPU模式]
       ↓
       timer.start()             [← 计时开始]
       dnnl_primitive_execute()  [← 被计时的操作]
       timer.stamp()             [← 计时结束]
```

### 代码位置

**入口函数**：`matmul.cpp:1088` - `doit()`
```cpp
int doit(...) {
    // 不计时的初始化操作
    init_memory_args(...);
    init_ref_memory_args(...);
    execute_and_wait(...);  // 第一次执行，用于验证
    check_correctness(...);
    
    // 性能测量（计时部分）
    return measure_perf(prb->ctx_exe, res, prim, args);
}
```

**性能测量函数**：`dnnl_common.cpp:611` - `measure_perf()`
- CPU（非DPCPP）→ `measure_perf_individual()` (dnnl_common.cpp:498)
- GPU/DPCPP → `measure_perf_aggregate()` (dnnl_common.cpp:513)

**核心计时逻辑**：`dnnl_common.cpp:498` - `measure_perf_individual()`
```cpp
inline int measure_perf_individual(...) {
    cold_cache_t cold_cache(dnnl_args, stream);
    t.reset();
    
    while (true) {
        if (!cold_cache.update_dnnl_args(dnnl_args)) break;
        
        t.start();  // ← 计时开始点
        DNN_SAFE(perf_func(stream, dnnl_args), WARN);  // ← 执行 dnnl_primitive_execute
        t.stamp();  // ← 计时结束点，记录本次执行时间
        
        if (should_stop(t)) break;
    }
}
```

**计时器实现**：`utils/timer.cpp`
- `timer_t::start()` - 记录开始时间（第58行）
- `timer_t::stamp()` - 记录结束时间并计算差值（第91行）

## 计时包含的操作

**只包含**：
- `dnnl_primitive_execute()` 的执行时间（矩阵乘法的实际计算）

**不包含**：
- 内存分配和初始化（`init_memory_args`）
- 数据填充（`fill_data`）
- 第一次执行（`execute_and_wait`，用于验证/预热）
- 正确性检查（`check_correctness`）
- 结果验证（`check_bitwise`）
- 内存映射/解映射（`execute_map_args`/`execute_unmap_args`）

## Cold Cache 机制

### 工作原理

benchdnn 使用 **Cold Cache** 来模拟真实场景中的缓存未命中：

1. **多缓冲区分配**（`utils/cold_cache.cpp:46`）
   - 分配多个内存缓冲区（CPU最多10000个）
   - 每个缓冲区大小足以容纳整个矩阵数据

2. **每次迭代切换缓冲区**（`utils/cold_cache.cpp:319`）
   ```cpp
   bool cold_cache_t::update_dnnl_args(...) {
       // 轮换使用不同的缓冲区
       dnnl_args[idx].memory = cache_[cc_counter_].m_;
       cc_counter_++;  // 下次使用下一个缓冲区
   }
   ```

3. **缓存未命中模拟**：
   - 第一次迭代可能使用已预热的缓冲区（对应 min 时间）
   - 后续迭代使用新缓冲区，导致 L1/L2/L3 缓存未命中（对应 avg 时间）

### 为什么 min < avg？

- **min 时间**：最佳缓存命中情况（数据已在缓存中）
- **avg 时间**：多次迭代的平均值（包含缓存未命中）
- min < avg 是正常现象，反映缓存未命中的真实场景

## 线程配置

### 默认行为

**默认使用所有可用 CPU 核心**：
- 从 `DNNL_VERBOSE` 输出可以看到 `nthr:192`（192个线程）
- 每个线程在独立的核心上运行
- 每个核心使用自己的 AMX/AVX 单元

### 限制为单核心

```bash
OMP_NUM_THREADS=1 benchdnn ...
```
- 输出会显示 `nthr:1`
- 只使用一个核心和一个 AMX/AVX 单元
- 用于测试单核心性能

### 性能差异

对于 `512×1024:1024×512` 矩阵：
- 多核心（192）：avg = 0.030ms
- 单核心（1）：avg = 0.252ms
- 加速比约 8.4 倍（而非 192 倍，受限于内存带宽等）

## 停止条件

在 `dnnl_common.hpp:252` - `should_stop()`：
```cpp
bool should_stop(const timer::timer_t &t) {
    return (fix_times_per_prb && t.times() >= fix_times_per_prb)
        || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
            && t.times() >= min_times_per_prb);
}
```

**默认参数**（`benchdnn.cpp:65-67`）：
- `min_times_per_prb`: 5（最少执行5次）
- `max_ms_per_prb`: 3000ms（最多3秒）
- `fix_times_per_prb`: 0（使用时间/次数条件）

## 小矩阵的特殊情况

对于极小矩阵（如 8×256:256×8）：
- **固定开销占主导**（~8ms）
- 计算时间可忽略（~9.3ns）
- benchdnn 的时间主要反映固定开销，而非计算性能

固定开销来源：
- Kernel 启动开销
- 内存管理开销
- 系统调用开销
- Cold Cache 机制开销
- 线程同步开销（即使矩阵很小）

## 时间统计

### min/avg/max 的含义

- **min(ms)**：所有迭代中最快的一次（最佳缓存命中）
- **avg(ms)**：所有迭代的平均值（真实平均性能）
- **max(ms)**：所有迭代中最慢的一次（最差缓存命中）

### 统计计算

在 `utils/timer.cpp:63` - `timer_t::stop()`：
```cpp
void timer_t::stop(int add_times, int64_t add_ticks, double add_ms) {
    ms_[mode_t::avg] += add_ms;  // 累加平均时间
    ms_[mode_t::sum] += add_ms;  // 累加总时间
    ms_[mode_t::min] = std::min(ms_[mode_t::min], d_ms);  // 最小值
    ms_[mode_t::max] = std::max(ms_[mode_t::max], d_ms);  // 最大值
    times_ += add_times;
}
```

最终 `avg = sum / times_`

## 关键代码文件

| 功能 | 文件 | 行号 |
|------|------|------|
| 主执行函数 | `matmul.cpp` | 1088 (`doit`) |
| 性能测量入口 | `dnnl_common.cpp` | 611 (`measure_perf`) |
| CPU模式计时 | `dnnl_common.cpp` | 498 (`measure_perf_individual`) |
| GPU模式计时 | `dnnl_common.cpp` | 513 (`measure_perf_aggregate`) |
| 计时器实现 | `utils/timer.cpp` | 58 (`start`), 91 (`stamp`) |
| Cold Cache | `utils/cold_cache.cpp` | 319 (`update_dnnl_args`) |
| 停止条件 | `dnnl_common.hpp` | 252 (`should_stop`) |

## 总结

**计时起点**：`timer_t::start()` 调用时（在 `measure_perf_individual` 中）
**计时终点**：`timer_t::stamp()` 调用时（在 `measure_perf_individual` 中）
**被计时的操作**：`dnnl_primitive_execute()` 的执行
**结果**：多次迭代的最小值、平均值、最大值

**重要提示**：
- avg(ms) 只包含 primitive 执行时间，不包含初始化等开销
- 默认使用所有核心，要测试单核心需设置 `OMP_NUM_THREADS=1`
- min < avg 是正常的，反映 Cold Cache 机制的效果

