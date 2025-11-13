#!/usr/bin/env python3
"""
大模型推理性能测试脚本
用于VTune性能分析，包括AMX利用率、MFU和缓存命中率
"""

import os
import sys
import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any

class ModelInferenceBenchmark:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化模型推理基准测试
        
        Args:
            model_path: 模型路径
            device: 运行设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.warmup_runs = 3
        self.benchmark_runs = 10
        
    def load_model(self):
        """加载模型和分词器"""
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # 使用float32确保CPU推理
                device_map="cpu" if self.device == "cpu" else "auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            print(f"Model loaded successfully. Model type: {type(self.model)}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def prepare_inputs(self, prompt: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """准备输入数据"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        if self.device == "cpu":
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        return inputs
    
    def warmup(self, prompt: str):
        """模型预热"""
        print("Starting model warmup...")
        inputs = self.prepare_inputs(prompt)
        
        with torch.no_grad():
            for i in range(self.warmup_runs):
                print(f"Warmup run {i+1}/{self.warmup_runs}")
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        print("Warmup completed.")
    
    def benchmark_inference(self, prompt: str, max_new_tokens: int = 100) -> Dict[str, Any]:
        """执行推理基准测试"""
        print(f"Starting benchmark with {self.benchmark_runs} runs...")
        print(f"Prompt: {prompt[:100]}...")
        
        inputs = self.prepare_inputs(prompt)
        input_length = inputs['input_ids'].shape[1]
        
        # 存储性能指标
        generation_times = []
        total_tokens = []
        throughputs = []
        
        with torch.no_grad():
            for i in range(self.benchmark_runs):
                print(f"Benchmark run {i+1}/{self.benchmark_runs}")
                
                # 记录开始时间
                start_time = time.time()
                
                # 生成文本
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # 记录结束时间
                end_time = time.time()
                
                # 计算指标
                generation_time = end_time - start_time
                output_length = outputs.shape[1]
                new_tokens = output_length - input_length
                
                generation_times.append(generation_time)
                total_tokens.append(new_tokens)
                throughputs.append(new_tokens / generation_time)
                
                print(f"  Generated {new_tokens} tokens in {generation_time:.3f}s")
                print(f"  Throughput: {new_tokens / generation_time:.2f} tokens/s")
        
        # 计算统计信息
        stats = {
            'avg_generation_time': np.mean(generation_times),
            'std_generation_time': np.std(generation_times),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'total_tokens_generated': sum(total_tokens),
            'input_length': input_length,
            'max_new_tokens': max_new_tokens
        }
        
        return stats
    
    def run_comprehensive_benchmark(self):
        """运行综合基准测试"""
        # 测试提示词
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology continues to evolve,",
            "The key to solving complex problems lies in",
            "Machine learning algorithms have revolutionized",
            "The intersection of science and technology"
        ]
        
        print("=" * 60)
        print("COMPREHENSIVE MODEL INFERENCE BENCHMARK")
        print("=" * 60)
        
        all_stats = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test Case {i+1}/{len(test_prompts)} ---")
            
            # 预热
            self.warmup(prompt)
            
            # 基准测试
            stats = self.benchmark_inference(prompt, max_new_tokens=100)
            all_stats.append(stats)
            
            # 打印结果
            print(f"\nResults for prompt {i+1}:")
            print(f"  Average generation time: {stats['avg_generation_time']:.3f} ± {stats['std_generation_time']:.3f} seconds")
            print(f"  Average throughput: {stats['avg_throughput']:.2f} ± {stats['std_throughput']:.2f} tokens/second")
            print(f"  Total tokens generated: {stats['total_tokens_generated']}")
        
        # 总体统计
        print("\n" + "=" * 60)
        print("OVERALL BENCHMARK RESULTS")
        print("=" * 60)
        
        avg_throughput = np.mean([s['avg_throughput'] for s in all_stats])
        avg_generation_time = np.mean([s['avg_generation_time'] for s in all_stats])
        total_tokens = sum([s['total_tokens_generated'] for s in all_stats])
        
        print(f"Average throughput across all tests: {avg_throughput:.2f} tokens/second")
        print(f"Average generation time: {avg_generation_time:.3f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        
        return all_stats

def main():
    parser = argparse.ArgumentParser(description="大模型推理性能基准测试")
    parser.add_argument("--model_path", type=str, 
                       default="/data/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="运行设备")
    parser.add_argument("--warmup_runs", type=int, default=3,
                       help="预热运行次数")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                       help="基准测试运行次数")
    
    args = parser.parse_args()
    
    # 创建基准测试实例
    benchmark = ModelInferenceBenchmark(args.model_path, args.device)
    benchmark.warmup_runs = args.warmup_runs
    benchmark.benchmark_runs = args.benchmark_runs
    
    # 加载模型
    benchmark.load_model()
    
    # 运行基准测试
    results = benchmark.run_comprehensive_benchmark()
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()
