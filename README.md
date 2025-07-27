# Kyrie12infer

🚀 **Kyrie12infer** 是一个高性能的大语言模型推理引擎，专为多GPU环境下的高效推理而设计。

## ✨ 特性

- 🔥 **高性能推理**：优化的CUDA内核和内存管理
- 🎯 **多GPU支持**：支持Tensor Parallelism分布式推理
- 🛠️ **易于使用**：简洁的API接口，快速上手
- 📊 **性能对比**：内置vLLM性能对比工具
- 🐳 **Docker支持**：完整的容器化部署方案
- ⚡ **内存优化**：高效的KV缓存和内存池管理

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/J12-Kyrie/Kyrie12infer.git
cd Kyrie12infer
pip install -e .
```

### 基本使用

```python
from kyrie12infer import LLM, SamplingParams

# 初始化模型
llm = LLM("./kyrie12infer/qwen3_0.6b/", tensor_parallel_size=2)

# 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 生成文本
prompts = ["Hello, Kyrie12infer."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated text: {output.outputs[0].text}")
```

## 🐳 Docker部署

### 单独运行Kyrie12infer

```bash
# 构建并启动容器
sudo docker compose up --build -d

# 查看日志
sudo docker logs Kyrie12infer
```

### 性能对比测试

运行Kyrie12infer vs vLLM性能对比：

```bash
# 一键启动对比测试
./run_comparison.sh

# 或手动运行
python3 benchmark_comparison.py
```

## 📊 性能表现

在Qwen3-0.6B模型上的性能测试结果：

| 引擎          | 总Token数   | 时间(s)  | 吞吐量(tok/s)         |
|---------------|-------------|----------|-----------------------|
| **Kyrie12infer** | 133,966     | 93.41    | **1434.13**           |
| vLLM          | 133,966     | 115.82   | 1156.42               |
| **性能提升**  | -           | **19.3%** | **24.1%**             |

## 🛠️ 项目结构

```
Kyrie12infer/
├── kyrie12infer/              # 核心推理引擎
│   ├── engine/               # 推理引擎核心
│   ├── layers/               # 神经网络层实现
│   ├── models/               # 模型定义
│   └── utils/                # 工具函数
├── bench.py                  # Kyrie12infer基准测试
├── bench_vllm.py            # vLLM基准测试
├── benchmark_comparison.py   # 性能对比脚本
├── docker-compose.yml       # Kyrie12infer容器配置
├── docker-compose.vllm.yml  # vLLM容器配置
├── Dockerfile               # Kyrie12infer镜像
├── Dockerfile.vllm          # vLLM镜像
└── run_comparison.sh        # 一键对比脚本
```

## 🔧 配置说明

### 环境变量

- `MASTER_ADDR`: 分布式训练主节点地址（默认：localhost）
- `MASTER_PORT`: 分布式训练端口（默认：29500）
- `NCCL_DEBUG`: NCCL调试级别（推荐：INFO）
- `NCCL_SOCKET_IFNAME`: 网络接口（Docker中推荐：eth0）

### GPU配置

```python
# 单GPU
llm = LLM(model_path, tensor_parallel_size=1)

# 多GPU（推荐）
llm = LLM(model_path, tensor_parallel_size=2)
```

## 📚 API文档

### LLM类

```python
class LLM:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        enforce_eager: bool = False
    )
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True
    ) -> List[RequestOutput]
```

### SamplingParams类

```python
class SamplingParams:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 16,
        ignore_eos: bool = False
    )
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢vLLM项目提供的优秀基础架构
- 感谢PyTorch和CUDA团队的底层支持
- 感谢所有贡献者的努力

## 📞 联系

- 项目链接：[https://github.com/J12-Kyrie/Kyrie12infer](https://github.com/J12-Kyrie/Kyrie12infer)
- 问题反馈：[Issues](https://github.com/J12-Kyrie/Kyrie12infer/issues)

---

⭐ 如果这个项目对你有帮助，请给我们一个Star！