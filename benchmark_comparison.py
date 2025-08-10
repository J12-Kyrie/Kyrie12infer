#!/usr/bin/env python3
"""
性能对比脚本：Kyrie12infer vs vLLM
运行两个推理引擎的基准测试并对比结果
"""

import subprocess
import time
import re
import sys
from typing import Dict, Optional


def run_benchmark(container_name: str, description: str) -> Optional[Dict[str, float]]:
    """
    运行指定容器的基准测试
    
    Args:
        container_name: Docker容器名称
        description: 测试描述
    
    Returns:
        包含测试结果的字典，如果失败则返回None
    """
    print(f"\n{'='*50}")
    print(f"开始运行 {description} 基准测试...")
    print(f"{'='*50}")
    
    try:
        # 运行容器中的基准测试
        result = subprocess.run(
            ["docker", "exec", container_name, "python", "bench_vllm.py" if "vllm" in container_name else "bench.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode != 0:
            print(f"❌ {description} 测试失败:")
            print(f"错误输出: {result.stderr}")
            return None
        
        # 解析输出结果
        output = result.stdout.strip()
        print(f"📊 {description} 输出:")
        print(output)
        
        # 提取性能数据
        pattern = r"Total: (\d+)tok, Time: ([\d.]+)s, Throughput: ([\d.]+)tok/s"
        match = re.search(pattern, output)
        
        if match:
            total_tokens = float(match.group(1))
            time_taken = float(match.group(2))
            throughput = float(match.group(3))
            
            return {
                "total_tokens": total_tokens,
                "time_taken": time_taken,
                "throughput": throughput
            }
        else:
            print(f"⚠️  无法解析 {description} 的输出结果")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 测试超时")
        return None
    except Exception as e:
        print(f"❌ {description} 测试出错: {e}")
        return None


def check_container_status(container_name: str) -> bool:
    """
    检查容器是否正在运行
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        return container_name in result.stdout
    except Exception:
        return False


def print_comparison(nano_result: Dict[str, float], vllm_result: Dict[str, float]):
    """
    打印对比结果
    """
    print(f"\n{'='*60}")
    print("🏆 性能对比结果")
    print(f"{'='*60}")
    
    print(f"{'指标':<20} {'Kyrie12infer':<15} {'vLLM':<15} {'差异':<15}")
    print("-" * 60)
    
    # 吞吐量对比
    nano_throughput = nano_result["throughput"]
    vllm_throughput = vllm_result["throughput"]
    throughput_diff = ((nano_throughput - vllm_throughput) / vllm_throughput) * 100
    
    print(f"{'吞吐量 (tok/s)':<20} {nano_throughput:<15.2f} {vllm_throughput:<15.2f} {throughput_diff:+.1f}%")
    
    # 时间对比
    nano_time = nano_result["time_taken"]
    vllm_time = vllm_result["time_taken"]
    time_diff = ((nano_time - vllm_time) / vllm_time) * 100
    
    print(f"{'执行时间 (s)':<20} {nano_time:<15.2f} {vllm_time:<15.2f} {time_diff:+.1f}%")
    
    # 总结
    print(f"\n{'='*60}")
    if nano_throughput > vllm_throughput:
        winner = "Kyrie12infer"
        advantage = throughput_diff
    else:
        winner = "vLLM"
        advantage = -throughput_diff
    
    print(f"🎯 结论: {winner} 在吞吐量上领先 {advantage:.1f}%")
    print(f"{'='*60}")


def main():
    print("🚀 Kyrie12infer vs vLLM 性能对比测试")
    print("确保两个容器都在运行中...")
    
    # 检查容器状态
    containers = {
        "Kyrie12infer": "Kyrie12infer",
        "vllm-qwen3": "vLLM"
    }
    
    for container_name, description in containers.items():
        if not check_container_status(container_name):
            print(f"❌ 容器 {container_name} ({description}) 未运行")
            print(f"请先启动容器: docker-compose up -d {container_name}")
            sys.exit(1)
        else:
            print(f"✅ 容器 {container_name} ({description}) 正在运行")
    
    # 运行基准测试
    nano_result = run_benchmark("Kyrie12infer", "Kyrie12infer")
    vllm_result = run_benchmark("vllm-qwen3", "vLLM")
    
    # 检查结果并对比
    if nano_result and vllm_result:
        print_comparison(nano_result, vllm_result)
    else:
        print("\n❌ 无法完成对比，某些测试失败")
        if not nano_result:
            print("- Kyrie12infer 测试失败")
        if not vllm_result:
            print("- vLLM 测试失败")


if __name__ == "__main__":
    main()