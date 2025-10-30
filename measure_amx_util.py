#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from typing import List, Tuple


def run_perf_for_command(command: List[str], extra_env: dict | None = None) -> Tuple[int, str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Measure both total core cycles and AMX busy cycles
    perf_cmd = [
        "perf",
        "stat",
        "-x",
        ",",
        "-e",
        "cycles,exe.amx_busy",
        "--",
        *command,
    ]

    proc = subprocess.run(perf_cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def parse_perf_stat(stderr_text: str) -> Tuple[int, int]:
    """Parse perf stat -x , output for cycles and exe.amx_busy.

    Returns: (cycles, amx_busy)
    """
    cycles = 0
    amx_busy = 0
    for line in stderr_text.splitlines():
        # perf -x , emits: value,unit,event, ...
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        value_str, _unit, event = parts[0], parts[1], parts[2]
        # Skip non-numeric values like <not supported>
        try:
            value = int(value_str.replace(" ", ""))
        except ValueError:
            continue
        if event == "cycles":
            cycles = value
        elif event == "exe.amx_busy":
            amx_busy = value
    return cycles, amx_busy


def compute_utilization(cycles: int, amx_busy: int) -> float:
    if cycles <= 0:
        return 0.0
    return (amx_busy / cycles) * 100.0


def cmd_mode(args: argparse.Namespace) -> int:
    command = shlex.split(args.command)
    rc, _out, err = run_perf_for_command(command, extra_env={
        "ONEDNN_MAX_CPU_ISA": "AVX512_CORE_AMX",
    })
    cycles, amx_busy = parse_perf_stat(err)
    util = compute_utilization(cycles, amx_busy)
    print("=== AMX Utilization (command mode) ===")
    print(f"Command: {args.command}")
    print(f"cycles: {cycles:,}")
    print(f"exe.amx_busy: {amx_busy:,}")
    print(f"AMX utilization: {util:.2f}%")
    return rc


def matmul_mode(args: argparse.Namespace) -> int:
    # Build a temporary python snippet to do pure matmul workload
    snippet = f"""
import torch
import os
os.environ['ONEDNN_MAX_CPU_ISA'] = 'AVX512_CORE_AMX'

dtype = torch.bfloat16 if '{args.dtype}'.lower() == 'bf16' else torch.float32
M, K, N = {args.m}, {args.k}, {args.n}
iters = {args.iters}

a = torch.randn(M, K, dtype=dtype, device='cpu')
b = torch.randn(K, N, dtype=dtype, device='cpu')

with torch.no_grad():
  with torch.amp.autocast('cpu', dtype=dtype):
    for i in range(iters):
      c = torch.mm(a, b)
"""

    tmp_file = "/tmp/_amx_matmul_bench.py"
    with open(tmp_file, "w") as f:
        f.write(snippet)

    rc, _out, err = run_perf_for_command([sys.executable, tmp_file], extra_env={})
    cycles, amx_busy = parse_perf_stat(err)
    util = compute_utilization(cycles, amx_busy)
    print("=== AMX Utilization (matmul mode) ===")
    print(f"Shape: ({args.m}, {args.k}) x ({args.k}, {args.n}), dtype={args.dtype}, iters={args.iters}")
    print(f"cycles: {cycles:,}")
    print(f"exe.amx_busy: {amx_busy:,}")
    print(f"AMX utilization: {util:.2f}%")
    return rc


def main():
    parser = argparse.ArgumentParser(description="Measure AMX utilization with perf (cycles vs exe.amx_busy)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_cmd = subparsers.add_parser("cmd", help="Measure AMX utilization for an arbitrary command")
    p_cmd.add_argument("--command", required=True, help="Command to run; quoted string")
    p_cmd.set_defaults(func=cmd_mode)

    p_mm = subparsers.add_parser("matmul", help="Measure AMX utilization for a pure matmul workload")
    p_mm.add_argument("--m", type=int, required=True)
    p_mm.add_argument("--k", type=int, required=True)
    p_mm.add_argument("--n", type=int, required=True)
    p_mm.add_argument("--iters", type=int, default=10)
    p_mm.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p_mm.set_defaults(func=matmul_mode)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()


