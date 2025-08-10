#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize_ = 32;
  int lane = tid % warpSize_;
  int warp_id = tid / warpSize_;
  for (int offset = warpSize_ / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  __shared__ float warpSums[32];
  if (lane == 0) warpSums[warp_id] = val;
  __syncthreads();
  if (warp_id == 0) {
    val = (tid < (blockDim.x + warpSize_ - 1) / warpSize_) ? warpSums[tid] : 0.0f;
    for (int offset = warpSize_ / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  } else {
    val = 0.0f;
  }
  return val;
}

__global__ void rmsnorm_kernel(const float* in, const float* weight, float* out,
                               int batch, int size, float eps) {
  const int bid = blockIdx.x;
  if (bid >= batch) return;
  const float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i];
    sum += x * x;
  }
  __shared__ float shared_val;
  sum = block_reduce(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  const float scale = rsqrtf(shared_val / static_cast<float>(size) + eps);
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i] * weight[i];
    block_out[i] = x * scale;
  }
}
} // anonymous namespace

at::Tensor rmsnorm_fwd_cuda(const at::Tensor& input, const at::Tensor& weight, double eps) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(input.dtype() == at::kFloat && weight.dtype() == at::kFloat, "dtype must be float32");
  TORCH_CHECK(input.size(-1) == weight.size(0), "weight must match hidden size");
  auto x = input.contiguous();
  auto w = weight.contiguous();
  const int64_t batch = x.numel() / x.size(-1);
  const int64_t size = x.size(-1);
  auto out = at::empty_like(x);
  const int block_size = 1024;
  const dim3 grid(batch);
  const dim3 block(block_size);
  rmsnorm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(),
      static_cast<int>(batch), static_cast<int>(size), static_cast<float>(eps));
  return out;
}


