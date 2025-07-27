#!/usr/bin/env python3
"""
简单的vLLM测试脚本
用于验证vLLM是否能正常加载和运行
"""

import os
import time
from vllm import LLM, SamplingParams


def main():
    print("🚀 开始vLLM简单测试...")
    
    try:
        # 模型路径
        model_path = "./nanovllm/qwen3_0.6b/"
        print(f"📂 加载模型: {model_path}")
        
        # 创建LLM实例
        llm = LLM(
            model=model_path,
            enforce_eager=False,
            max_model_len=4096,
            tensor_parallel_size=2
        )
        print("✅ 模型加载成功")
        
        # 简单的推理测试
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=50
        )
        
        print("🔄 开始推理测试...")
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        end_time = time.time()
        
        print("✅ 推理完成")
        print(f"⏱️  推理时间: {end_time - start_time:.2f}秒")
        
        # 显示结果
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📝 输入: {prompt}")
            print(f"📤 输出: {generated_text}")
        
        print("\n🎉 vLLM测试成功完成！")
        
    except Exception as e:
        print(f"❌ vLLM测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)