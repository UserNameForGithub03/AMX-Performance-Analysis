#!/usr/bin/env python3
"""
简化的AMX优化模型推理脚本
只进行模型加载和推理，不进行性能测试
使用Intel Extension for PyTorch (IPEX) 和 BF16 精度来触发AMX指令
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def load_model(model_path: str, device: str = "cpu", use_bf16: bool = True):
    """
    加载模型和分词器，并应用AMX优化
    
    Args:
        model_path: 模型路径
        device: 运行设备 (cpu/cuda)
        use_bf16: 是否使用BF16精度来触发AMX
    
    Returns:
        model: 加载的模型
        tokenizer: 分词器
    """
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    print(f"Precision: {'BF16' if use_bf16 and IPEX_AVAILABLE else 'FP32'}")
    print(f"AMX Optimization: {'Enabled' if use_bf16 and IPEX_AVAILABLE else 'Disabled'}")
    print(f"IPEX Available: {IPEX_AVAILABLE}")
    
    try:
        # 加载分词器
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        print("Loading model...")
        use_bf16_actual = use_bf16 and IPEX_AVAILABLE
        
        if use_bf16_actual:
            # 使用BF16精度加载模型以触发AMX
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 应用IPEX优化
            model = ipex.optimize(
                model.eval(), 
                dtype=torch.bfloat16, 
                inplace=True
            )
            print("Model optimized with IPEX for AMX acceleration")
        else:
            # 标准FP32加载
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        if device == "cpu":
            model = model.to("cpu")
        
        print(f"Model loaded successfully. Model type: {type(model)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def inference(model, tokenizer, prompt: str, max_new_tokens: int = 150, use_bf16: bool = True):
    """
    执行模型推理
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        prompt: 输入提示词
        max_new_tokens: 最大生成token数
        use_bf16: 是否使用BF16精度
    """
    print("\n" + "=" * 60)
    print("MODEL INFERENCE")
    print("=" * 60)
    print(f"Prompt: {prompt[:100]}...")
    
    # 准备输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    input_length = inputs['input_ids'].shape[1]
    
    print(f"Input length: {input_length} tokens")
    print("Generating response...")
    
    # 执行推理
    use_bf16_actual = use_bf16 and IPEX_AVAILABLE
    
    with torch.no_grad():
        if use_bf16_actual:
            with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
    
    # 解码输出
    output_length = outputs.shape[1]
    new_tokens = output_length - input_length
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nGenerated {new_tokens} new tokens")
    # print(f"Total output length: {output_length} tokens")
    # print("\n" + "-" * 60)
    # print("GENERATED TEXT:")
    # print("-" * 60)
    # print(generated_text)
    # print("-" * 60)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="AMX优化的模型推理脚本")
    parser.add_argument("--model_path", type=str, 
                       default="/data/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="运行设备")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                       help="最大生成token数")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                       help="使用BF16精度启用AMX优化")
    parser.add_argument("--use_fp32", action="store_true", default=False,
                       help="使用FP32精度（禁用AMX优化）")
    parser.add_argument("--num_runs", type=int, default=2,
                       help="推理执行次数")
    # /data/share/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
    args = parser.parse_args()
    
    # 如果指定了FP32，则禁用BF16
    if args.use_fp32:
        args.use_bf16 = False
    
    # 测试提示词（使用与原文件相同的prompt）
    test_prompt = "The future of artificial intelligence is rapidly unfolding before our eyes, bringing with it unprecedented opportunities and challenges that will reshape every aspect of human society. As we stand at this technological crossroads, researchers and engineers worldwide are developing increasingly sophisticated AI systems that can process vast amounts of data, recognize complex patterns, and make decisions with remarkable accuracy. These advancements span multiple domains including natural language processing, computer vision, robotics, and autonomous systems. The integration of machine learning algorithms with modern computing infrastructure has enabled breakthroughs that were once considered science fiction, from self-driving vehicles to medical diagnosis systems that can detect diseases earlier than human experts"
    
    # 加载模型
    model, tokenizer = load_model(args.model_path, args.device, use_bf16=args.use_bf16)
    
    # 执行推理
    for i in range(args.num_runs):
        generated_text = inference(
            model, 
            tokenizer, 
            test_prompt, 
            max_new_tokens=args.max_new_tokens,
            use_bf16=args.use_bf16
        )
        # print(f"Run {i+1} completed successfully!")
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()

