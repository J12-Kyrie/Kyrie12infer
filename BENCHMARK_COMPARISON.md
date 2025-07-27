# Kyrie12infer vs vLLM 性能对比测试

本文档介绍如何使用提供的工具对比 Kyrie12infer 和 vLLM 在 Qwen3-0.6B 模型上的推理性能。

## 📁 文件说明

### 核心文件
- `bench.py` - Kyrie12infer 基准测试脚本
- `bench_vllm.py` - vLLM 基准测试脚本
- `Dockerfile` - Kyrie12infer Docker 配置
- `Dockerfile.vllm` - vLLM Docker 配置
- `docker-compose.yml` - Kyrie12infer 容器编排
- `docker-compose.vllm.yml` - vLLM 容器编排

### 对比工具
- `benchmark_comparison.py` - 性能对比脚本
- `run_comparison.sh` - 一键启动对比测试
- `BENCHMARK_COMPARISON.md` - 本说明文档

## 🚀 快速开始

### 方法一：一键运行（推荐）

```bash
# 运行完整的对比测试
./run_comparison.sh
```

这个脚本会自动：
1. 构建两个 Docker 镜像
2. 启动两个容器
3. 等待模型加载
4. 运行性能对比测试
5. 显示详细的对比结果

### 方法二：手动运行

#### 1. 启动 Kyrie12infer 容器
```bash
sudo docker compose up --build -d
```

#### 2. 启动 vLLM 容器
```bash
sudo docker compose -f docker-compose.vllm.yml up --build -d
```

#### 3. 等待容器启动完成
```bash
# 检查容器状态
docker ps

# 查看启动日志
docker logs Kyrie12infer
docker logs vllm-qwen3
```

#### 4. 运行对比测试
```bash
python3 benchmark_comparison.py
```

## 📊 测试配置

两个基准测试使用相同的参数确保公平对比：

- **模型**: Qwen3-0.6B
- **序列数量**: 256
- **最大输入长度**: 1024 tokens
- **最大输出长度**: 1024 tokens
- **并行度**: tensor_parallel_size=2 (使用2个GPU)
- **温度**: 0.6
- **最大模型长度**: 4096 tokens

## 🔧 环境配置

### GPU 要求
- 至少 2 个 NVIDIA GPU
- CUDA 12.1 支持
- 足够的 GPU 内存加载 Qwen3-0.6B 模型

### Docker 配置
两个容器都配置了：
- NVIDIA GPU 运行时
- NCCL 多GPU通信
- 2GB 共享内存
- 相同的环境变量

### 端口映射
- Kyrie12infer: 8000 → 8000
- vLLM: 8000 → 8001

## 📈 结果解读

对比脚本会显示以下指标：

1. **吞吐量 (tok/s)**: 每秒处理的 token 数量
2. **执行时间 (s)**: 完成测试所需的时间
3. **性能差异 (%)**: 两个引擎之间的相对性能差异

### 示例输出
```
============================================================
🏆 性能对比结果
============================================================
指标                 Kyrie12infer       vLLM            差异           
------------------------------------------------------------
吞吐量 (tok/s)        7738.67         6234.56         +24.1%
执行时间 (s)          16.45           20.42           -19.4%

============================================================
🎯 结论: Kyrie12infer 在吞吐量上领先 24.1%
============================================================
```

## 🛠️ 故障排除

### 容器启动失败
```bash
# 查看详细错误日志
docker logs Kyrie12infer
docker logs vllm-qwen3

# 检查 GPU 可用性
nvidia-smi

# 重新构建镜像
docker-compose build --no-cache
docker-compose -f docker-compose.vllm.yml build --no-cache
```

### NCCL 通信错误
如果遇到 NCCL 相关错误，检查：
1. GPU 数量是否足够
2. CUDA 版本兼容性
3. 网络接口配置

### 内存不足
如果遇到 OOM 错误：
1. 减少 `tensor_parallel_size`
2. 降低 `max_model_len`
3. 减少测试序列数量

## 🧹 清理

```bash
# 停止所有容器
sudo docker compose down
sudo docker compose -f docker-compose.vllm.yml down

# 删除镜像（可选）
sudo docker rmi kyrie12infer-Kyrie12infer
sudo docker rmi kyrie12infer-vllm

# 清理未使用的资源
sudo docker system prune
```

## 📝 自定义测试

### 修改测试参数
编辑 `bench.py` 和 `bench_vllm.py` 中的参数：
- `num_seqs`: 序列数量
- `max_input_len`: 最大输入长度
- `max_ouput_len`: 最大输出长度
- `tensor_parallel_size`: GPU 并行度

### 使用不同模型
1. 将模型文件放在 `kyrie12infer/` 目录下
2. 修改脚本中的模型路径
3. 更新 Docker 卷映射

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这些对比工具！