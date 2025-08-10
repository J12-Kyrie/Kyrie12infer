#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize_ = 32;
  int lane = tid % warpSize_;
  int warp_id = tid / warpSize_;

  #pragma unroll
  for (int offset = warpSize_ / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  __shared__ float warpSums[32];
  if (lane == 0) {
    warpSums[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = (tid < blockDim.x / 32) ? warpSums[tid] : 0.0f;
    #pragma unroll
    for (int offset = warpSize_ / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
  }
  return val;
}

__global__ void reduce_kernel(const float* in, float* out, int n) {
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    sum += in[i];
  }
  sum = block_reduce(sum);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}
} // anonymous namespace

at::Tensor reduce_sum_cuda(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
  TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
  auto contig = x.contiguous();
  const int64_t n = contig.numel();
  const int block_size = 1024;
  const int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
  auto tmp = at::empty({num_blocks}, contig.options());
  reduce_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
      contig.data_ptr<float>(), tmp.data_ptr<float>(), static_cast<int>(n));
  // final reduce on device using same kernel
  auto out = at::empty({1}, contig.options());
  reduce_kernel<<<1, num_blocks, 0, at::cuda::getCurrentCUDAStream()>>>(
      tmp.data_ptr<float>(), out.data_ptr<float>(), num_blocks);
  return out[0];
}


