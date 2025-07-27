#!/bin/bash

# nano-vLLM vs vLLM 性能对比测试启动脚本

set -e

echo "🚀 nano-vLLM vs vLLM 性能对比测试"
echo "================================="

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker 未运行，请先启动 Docker"
    exit 1
fi

echo "📦 构建和启动容器..."

# 停止现有容器（如果存在）
echo "🛑 停止现有容器..."
sudo docker compose down 2>/dev/null || true
sudo docker compose -f docker-compose.vllm.yml down 2>/dev/null || true

# 构建并启动 nano-vLLM 容器
echo "🔨 构建并启动 nano-vLLM 容器..."
sudo docker compose up --build -d

# 构建并启动 vLLM 容器
echo "🔨 构建并启动 vLLM 容器..."
sudo docker compose -f docker-compose.vllm.yml up --build -d

echo "⏳ 等待容器启动完成..."
sleep 10

# 检查容器状态
echo "📋 检查容器状态..."
if docker ps --filter "name=nano-vllm" --format "{{.Names}}" | grep -q "nano-vllm"; then
    echo "✅ nano-vLLM 容器运行中"
else
    echo "❌ nano-vLLM 容器启动失败"
    docker logs nano-vllm
    exit 1
fi

if docker ps --filter "name=vllm-qwen3" --format "{{.Names}}" | grep -q "vllm-qwen3"; then
    echo "✅ vLLM 容器运行中"
else
    echo "❌ vLLM 容器启动失败"
    docker logs vllm-qwen3
    exit 1
fi

echo "🎯 开始性能对比测试..."
echo "================================="

# 给容器更多时间初始化
echo "⏳ 等待模型加载完成..."
sleep 30

# 运行对比测试
python3 benchmark_comparison.py

echo "\n🏁 测试完成！"
echo "\n📊 查看详细日志:"
echo "  nano-vLLM: docker logs nano-vllm"
echo "  vLLM: docker logs vllm-qwen3"

echo "\n🛑 停止容器:"
echo "  docker-compose down"
echo "  docker-compose -f docker-compose.vllm.yml down"