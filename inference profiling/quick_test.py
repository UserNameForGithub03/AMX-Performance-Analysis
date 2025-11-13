#!/usr/bin/env python3
"""
快速测试脚本 - 用于验证模型加载和基本推理功能
"""

import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_test(model_path):
    """快速测试模型加载和推理"""
    print(f"Testing model: {model_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU cores: {torch.get_num_threads()}")
    print(f"Device available: {torch.cuda.is_available()}")
    
    try:
        # 加载分词器
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载模型
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 简单推理测试
        prompt = "The future of AI is"
        print(f"\nTesting inference with prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        print("\nQuick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during quick test: {e}")
        return False

if __name__ == "__main__":
    model_path = "/data/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    success = quick_test(model_path)
    sys.exit(0 if success else 1)
