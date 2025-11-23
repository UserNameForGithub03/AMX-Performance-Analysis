#!/usr/bin/env python3
"""
启用AMX优化的大模型推理性能测试脚本
使用Intel Extension for PyTorch (IPEX) 和 BF16 精度来触发AMX指令
"""

import os
import sys
import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any

# 设置环境变量以启用AMX
os.environ['ONEDNN_VERBOSE'] = '0'
os.environ['ONEDNN_MAX_CPU_ISA'] = 'AVX512_CORE_AMX'
os.environ['OMP_NUM_THREADS'] = '96'  # 使用所有物理核心
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

try:
    import intel_extension_for_pytorch as ipex
    print(f"IPEX version: {ipex.__version__}")
    IPEX_AVAILABLE = True
except ImportError:
    print("Warning: Intel Extension for PyTorch not available. Using standard PyTorch.")
    IPEX_AVAILABLE = False

class AMXOptimizedModelInferenceBenchmark:
    def __init__(self, model_path: str, device: str = "cpu", use_bf16: bool = True):
        """
        初始化AMX优化的模型推理基准测试
        
        Args:
            model_path: 模型路径
            device: 运行设备 (cpu/cuda)
            use_bf16: 是否使用BF16精度来触发AMX
        """
        self.model_path = model_path
        self.device = device
        self.use_bf16 = use_bf16 and IPEX_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.warmup_runs = 1
        self.benchmark_runs = 5
        
        print(f"AMX Optimization: {'Enabled' if self.use_bf16 else 'Disabled'}")
        print(f"IPEX Available: {IPEX_AVAILABLE}")
        
    def load_model(self):
        """加载模型和分词器，并应用AMX优化"""
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Precision: {'BF16' if self.use_bf16 else 'FP32'}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            if self.use_bf16:
                # 使用BF16精度加载模型以触发AMX
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # 应用IPEX优化
                self.model = ipex.optimize(
                    self.model.eval(), 
                    dtype=torch.bfloat16, 
                    inplace=True
                )
                print("Model optimized with IPEX for AMX acceleration")
            else:
                # 标准FP32加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
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
                if self.use_bf16:
                    with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                        _ = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            temperature=1.0,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
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
                if self.use_bf16:
                    with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=1.0,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                else:
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
        # 测试提示词 (10倍长度版本 - 流畅文本)
        test_prompts = [
            "The future of artificial intelligence is rapidly unfolding before our eyes, bringing with it unprecedented opportunities and challenges that will reshape every aspect of human society. As we stand at this technological crossroads, researchers and engineers worldwide are developing increasingly sophisticated AI systems that can process vast amounts of data, recognize complex patterns, and make decisions with remarkable accuracy. These advancements span multiple domains including natural language processing, computer vision, robotics, and autonomous systems. The integration of machine learning algorithms with modern computing infrastructure has enabled breakthroughs that were once considered science fiction, from self-driving vehicles to medical diagnosis systems that can detect diseases earlier than human experts",
            
            "In a world where technology continues to evolve at an unprecedented pace, we are witnessing a fundamental transformation in how humans interact with machines and process information. Digital innovation has become the cornerstone of modern civilization, driving economic growth, scientific discovery, and social progress across all continents. The convergence of cloud computing, mobile connectivity, and advanced analytics has created an ecosystem where data flows seamlessly between devices, enabling real-time decision-making and collaborative problem-solving on a global scale. From smart cities that optimize energy consumption to personalized healthcare solutions that predict and prevent diseases, technological advancement is revolutionizing traditional industries while creating entirely new sectors",
            
            "The key to solving complex problems lies in our ability to combine systematic analytical thinking with creative innovation and interdisciplinary collaboration. Throughout history, humanity's greatest achievements have emerged from the synthesis of diverse perspectives, rigorous scientific methodology, and the courage to challenge conventional wisdom. Modern problem-solving requires not only technical expertise but also emotional intelligence, cultural awareness, and ethical consideration. As we face challenges ranging from climate change to global health crises, the importance of developing comprehensive solutions that address root causes rather than symptoms has never been more critical. Educational institutions and research organizations worldwide are adapting their curricula to prepare",
            
            "Machine learning algorithms have revolutionized the way we process and interpret data, enabling computers to learn from experience without being explicitly programmed for every possible scenario. These sophisticated mathematical models, inspired by biological neural networks, can identify intricate patterns in massive datasets that would be impossible for humans to detect manually. Applications of machine learning now permeate our daily lives, from recommendation systems that predict our preferences to fraud detection algorithms that protect financial transactions. Deep learning, a subset of machine learning using multilayered neural networks, has achieved superhuman performance in tasks such as image classification, speech recognition, and strategic game playing, fundamentally changing fields ranging from medical diagnostics",
            
            "The intersection of science and technology creates a dynamic environment where theoretical discoveries rapidly translate into practical applications that benefit society. This symbiotic relationship between pure research and applied engineering has accelerated innovation cycles, allowing breakthrough concepts to move from laboratory experiments to commercial products in increasingly shorter timeframes. Interdisciplinary fields such as bioinformatics, quantum computing, and nanotechnology exemplify how crossing traditional academic boundaries can unlock new frontiers of knowledge and capability. Government funding agencies, private corporations, and academic institutions are increasingly recognizing the value of supporting collaborative research initiatives that bring together experts from multiple disciplines to tackle humanity's most pressing challenges in healthcare, environmental sustainability, and information"
        ]
        
        print("=" * 60)
        print("AMX-OPTIMIZED MODEL INFERENCE BENCHMARK")
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
        print(f"AMX Optimization: {'Enabled (BF16)' if self.use_bf16 else 'Disabled (FP32)'}")
        
        return all_stats

def main():
    parser = argparse.ArgumentParser(description="AMX优化的大模型推理性能基准测试")
    parser.add_argument("--model_path", type=str, 
                       default="/data/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="运行设备")
    parser.add_argument("--warmup_runs", type=int, default=3,
                       help="预热运行次数")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                       help="基准测试运行次数")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                       help="使用BF16精度启用AMX优化")
    parser.add_argument("--use_fp32", action="store_true", default=False,
                       help="使用FP32精度（禁用AMX优化）")
    
    args = parser.parse_args()
    
    # 如果指定了FP32，则禁用BF16
    if args.use_fp32:
        args.use_bf16 = False
    
    # 创建基准测试实例
    benchmark = AMXOptimizedModelInferenceBenchmark(
        args.model_path, 
        args.device, 
        use_bf16=args.use_bf16
    )
    benchmark.warmup_runs = args.warmup_runs
    benchmark.benchmark_runs = args.benchmark_runs
    
    # 加载模型
    benchmark.load_model()
    
    # 运行基准测试
    results = benchmark.run_comprehensive_benchmark()
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()
