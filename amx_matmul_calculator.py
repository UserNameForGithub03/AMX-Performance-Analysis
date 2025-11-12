#!/usr/bin/env python3
"""
AMX矩阵乘法操作数和性能计算工具

用于计算：
1. 矩阵乘法所需的操作数（OPs）
2. 基于CPU频率和AMX每周期操作数的理论计算时间
"""

import argparse
import sys
import subprocess
import re
import os
from typing import Dict, Optional, Tuple

# AMX每周期操作数（基于Intel文档）
AMX_OPS_PER_CYCLE = {
    'bf16': 1024,   # BF16: 每个cycle约1024次乘加操作
    'int8': 2048,   # INT8: 每个cycle约2048次乘加操作
    'fp32': 1024,   # FP32: 与BF16相同
}

DEFAULT_CPU_FREQ_GHZ = 3.50  #因为观察到AMX运行的时候CPU的频率在3.5GHz左右
BENCHDNN_PATH = "/data/wangjiaqi/oneDNN/build/tests/benchdnn/benchdnn"


def calculate_matmul_ops(M: int, K: int, N: int) -> Dict[str, int]:
    """
    计算矩阵乘法C = A * B所需的操作数
    
    Args:
        M: A矩阵的行数
        K: A矩阵的列数（也是B矩阵的行数）
        N: B矩阵的列数
    
    Returns:
        包含操作数信息的字典
    """
    # 标准计算：每个C[i][j]需要K次乘加运算
    total_elements = M * N
    multiply_add_ops = total_elements * K  # 乘加操作数（乘加 = 1 OP）
    flops = multiply_add_ops * 2  # FLOPS（乘加 = 2 FLOPS）
    
    return {
        'multiply_add_ops': multiply_add_ops,
        'flops': flops,
        'total_elements': total_elements,
        'operations_per_element': K
    }


def warmup_cpu(benchdnn_path: str, warmup_iterations: int = 3) -> None:
    """
    CPU预热：通过运行小型benchdnn计算来预热CPU，提高频率
    
    Args:
        benchdnn_path: benchdnn工具路径
        warmup_iterations: 预热迭代次数（默认: 3）
    """
    # 使用小型矩阵进行预热计算
    warmup_M, warmup_K, warmup_N = 256, 256, 256
    warmup_dtype = 'bf16'
    
    cmd = build_benchdnn_command(warmup_M, warmup_K, warmup_N, warmup_dtype, benchdnn_path)
    
    env = os.environ.copy()
    env['DNNL_VERBOSE'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    
    print(f"   正在进行CPU预热 ({warmup_iterations}次)...")
    for i in range(warmup_iterations):
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
        except Exception:
            pass  # 忽略预热过程中的错误
    print(f"   CPU预热完成")


def get_dtype_for_benchdnn(dtype: str) -> str:
    """将内部数据类型转换为benchdnn格式"""
    dtype_map = {
        'bf16': 'bf16',
        'fp32': 'f32',
        'int8': 's8',
    }
    return dtype_map.get(dtype, 'bf16')


def build_benchdnn_command(M: int, K: int, N: int, dtype: str, benchdnn_path: str) -> list:
    """
    构建benchdnn命令
    
    Args:
        M, K, N: 矩阵维度
        dtype: 数据类型
        benchdnn_path: benchdnn工具路径
    
    Returns:
        命令列表
    """
    dt_benchdnn = get_dtype_for_benchdnn(dtype)
    # 格式: MxK:KxN -> A矩阵和B矩阵的维度
    problem = f"{M}x{K}:{K}x{N}"
    
    cmd = [
        benchdnn_path,
        "--matmul",
        "--mode=P",
        f"--dt={dt_benchdnn}:{dt_benchdnn}:{dt_benchdnn}",
        "--stag=any",
        "--wtag=any",
        "--dtag=any",
        problem
    ]
    return cmd


def run_benchdnn(M: int, K: int, N: int, dtype: str, benchdnn_path: str, do_warmup: bool = True) -> Optional[float]:
    """
    运行benchdnn并提取平均耗时（ms）
    
    Args:
        M, K, N: 矩阵维度
        dtype: 数据类型
        benchdnn_path: benchdnn工具路径
        do_warmup: 是否进行CPU预热（默认: True）
    
    Returns:
        平均耗时（ms），如果失败则返回None
    """
    cmd = build_benchdnn_command(M, K, N, dtype, benchdnn_path)
    
    env = os.environ.copy()
    dnnl_isa = 'AVX512_CORE_AMX'
    env['DNNL_MAX_CPU_ISA'] = dnnl_isa
    env['DNNL_VERBOSE'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    
    # 真实预热：在正式测量前先运行一次相同的benchdnn命令以提升CPU频率
    if do_warmup:
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )
        except Exception:
            pass

    # 打印执行的命令和环境变量
    print(f"\n   执行的benchdnn命令:")
    cmd_str = ' '.join(f"'{arg}'" if ' ' in arg else arg for arg in cmd)
    print(f"     DNNL_MAX_CPU_ISA={dnnl_isa} {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode != 0:
            print(f"警告: benchdnn命令执行失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"错误信息: {result.stderr[:200]}")
            return None
        
        # 从输出中提取 avg(ms) 值
        # 格式: "total perf: min(ms):4.06909 avg(ms):8.84451"
        output = result.stdout + result.stderr
        match = re.search(r'avg\(ms\):([\d.]+)', output)
        if match:
            avg_ms = float(match.group(1))
            return avg_ms
        else:
            print(f"警告: 无法从benchdnn输出中提取avg(ms)")
            print(f"输出预览: {output[-500:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print("警告: benchdnn命令执行超时")
        return None
    except FileNotFoundError:
        print(f"警告: 找不到benchdnn工具: {benchdnn_path}")
        return None
    except Exception as e:
        print(f"警告: 运行benchdnn时出错: {e}")
        return None


def calculate_theoretical_time(
    M: int, K: int, N: int,
    cpu_frequency_ghz: float,
    dtype: str = 'bf16'
) -> Dict[str, float]:
    """
    计算理论计算时间
    
    Args:
        M, K, N: 矩阵维度
        cpu_frequency_ghz: CPU频率（GHz）
        dtype: 数据类型
    
    Returns:
        理论时间信息
    """
    # 计算操作数（乘加算作两个ops）
    ops_info = calculate_matmul_ops(M, K, N)
    total_ops = ops_info['flops']
    
    # 获取每周期操作数
    ops_per_cycle = AMX_OPS_PER_CYCLE.get(dtype, AMX_OPS_PER_CYCLE['bf16'])
    
    # 基于操作数计算
    cycles_needed = total_ops / ops_per_cycle
    time_sec = cycles_needed / (cpu_frequency_ghz * 1e9)
    
    return {
        'total_ops': total_ops,
        'ops_per_cycle': ops_per_cycle,
        'cpu_frequency_ghz': cpu_frequency_ghz,
        'cycles_needed': cycles_needed,
        'time_sec': time_sec,
        'time_ms': time_sec * 1000,
    }


def print_analysis(M: int, K: int, N: int, cpu_frequency_ghz: float, dtype: str = 'bf16', 
                   run_benchmark: bool = True, benchdnn_path: str = BENCHDNN_PATH):
    """打印详细的分析报告"""
    print("=" * 80)
    print(f"矩阵维度: A({M}×{K}) × B({K}×{N}) = C({M}×{N})")
    print(f"数据类型: {dtype.upper()}")
    print(f"CPU频率: {cpu_frequency_ghz:.2f} GHz")
    print( "-" * 80)
    
    # 1. 操作数计算
    print("1. 操作数（OPs）计算:")
    ops_info = calculate_matmul_ops(M, K, N)
    print(f"   - C矩阵共有 {M} × {N} = {ops_info['total_elements']:,} 个元素")
    print(f"   - 每个元素需要 {K} 次乘加运算")
    print(f"   - 总乘加操作数: {ops_info['multiply_add_ops']:,} ops")
    print(f"   - 总FLOPS: {ops_info['flops']:,} FLOPS")
    
    # 2. AMX性能参数
    print("-" * 80)
    print("2. AMX每周期操作数:")
    ops_per_cycle = AMX_OPS_PER_CYCLE.get(dtype, AMX_OPS_PER_CYCLE['bf16'])
    print(f"   - 每周期操作数: {ops_per_cycle:,} ops/cycle ({dtype.upper()})")
    
    # 3. 理论计算时间
    print("-" * 80)
    print("3. 理论计算时间估算:")
    time_info = calculate_theoretical_time(M, K, N, cpu_frequency_ghz, dtype)
    
    print(f"   - 总操作数: {time_info['total_ops']:,} ops")
    print(f"   - 需要的cycles: {time_info['cycles_needed']:,.0f} cycles")
    print(f"   - 理论时间: {time_info['time_ms']:.4f} ms ({time_info['time_sec']:.6f} 秒)")
    
    # 性能指标
    total_flops = ops_info['flops']
    gflops = total_flops / time_info['time_sec'] / 1e9 if time_info['time_sec'] > 0 else 0
    peak_gflops = ops_per_cycle * 2 * cpu_frequency_ghz / 1e3  # 每周期FLOPS × 频率
    
    print(f"\n   - 理论吞吐量: {gflops:.2f} GFLOPS")
    print(f"   - AMX理论峰值吞吐量: {peak_gflops:.2f} GFLOPS")

    # 4. 运行benchdnn获取实际性能
    actual_time_ms = None
    efficiency = None
    
    if run_benchmark:
        print("-" * 80)
        print("4. 实际性能测试 (benchdnn):")
        print("   正在运行benchdnn...")
        
        actual_time_ms = run_benchdnn(M, K, N, dtype, benchdnn_path)
        
        if actual_time_ms is not None:
            print(f"   - 实际耗时: {actual_time_ms:.4f} ms")
            
            # 计算效率 = 理论时间 / 实际时间
            if actual_time_ms > 0:
                efficiency = (time_info['time_ms'] / actual_time_ms) * 100
                print(f"   - 效率: {efficiency:.2f}% (理论时间/实际时间 × 100%)")
                
                # 实际吞吐量
                actual_gflops = total_flops / (actual_time_ms / 1000.0) / 1e9
                print(f"   - 实际吞吐量: {actual_gflops:.2f} GFLOPS")
                
                print(f"\n   性能对比:")
                print(f"     理论时间: {time_info['time_ms']:.4f} ms")
                print(f"     实际时间: {actual_time_ms:.4f} ms")
                print(f"     效率比率: {efficiency:.2f}%")
            else:
                print("   警告: 实际时间为0，无法计算效率")
        else:
            print("   无法获取实际性能数据（benchdnn执行失败或未找到）")
  
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="计算AMX矩阵乘法的操作数和理论计算时间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算1024×1024矩阵乘法（自动获取CPU频率）
  python amx_matmul_calculator.py --M 1024 --K 1024 --N 1024
  
  # 指定CPU频率和数据类型
  python amx_matmul_calculator.py --M 1024 --K 1024 --N 1024 \\
      --freq 3.5 --dtype bf16
        """
    )
    
    parser.add_argument('--M', type=int, default=1024, help='矩阵A的行数（默认: 1024）')
    parser.add_argument('--K', type=int, default=1024, help='矩阵A的列数/B的行数（默认: 1024）')
    parser.add_argument('--N', type=int, default=1024, help='矩阵B的列数（默认: 1024）')
    parser.add_argument('--freq', '--frequency', type=float, default=None,
                       help=f'CPU频率（GHz）。如果不指定，使用固定值{DEFAULT_CPU_FREQ_GHZ} GHz')
    parser.add_argument('--dtype', choices=['bf16', 'int8', 'fp32'], default='bf16',
                       help='数据类型（默认: bf16）')
    parser.add_argument('--no-benchmark', action='store_true',
                       help='跳过benchdnn实际性能测试')
    parser.add_argument('--benchdnn-path', type=str, default=BENCHDNN_PATH,
                       help=f'benchdnn工具路径（默认: {BENCHDNN_PATH}）')
    
    args = parser.parse_args()
    
    # 固定使用4.0GHz作为CPU频率
    cpu_freq = args.freq if args.freq is not None else DEFAULT_CPU_FREQ_GHZ
    print(f"使用CPU频率: {cpu_freq:.2f} GHz（固定值）")
    
    print()  # 空行分隔
    
    # 打印分析结果
    print_analysis(args.M, args.K, args.N, cpu_freq, args.dtype, 
                   run_benchmark=not args.no_benchmark, benchdnn_path=args.benchdnn_path)


if __name__ == "__main__":
    main()
