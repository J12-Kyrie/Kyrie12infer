#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= N || gy >= M) return;
  float tmp = 0.f;
  for (int i = 0; i < K; ++i) tmp += A[gy * K + i] * B[i * N + gx];
  C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}
}

at::Tensor sgemm_cuda(const at::Tensor& A, const at::Tensor& B, double alpha, double beta) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA");
  TORCH_CHECK(A.dtype() == at::kFloat && B.dtype() == at::kFloat, "A/B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "A.cols must equal B.rows");
  auto a = A.contiguous();
  auto b = B.contiguous();
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  auto c = at::empty({M, N}, a.options());
  dim3 threads(32, 32);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  sgemm_naive_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, static_cast<float>(alpha), a.data_ptr<float>(), b.data_ptr<float>(), static_cast<float>(beta), c.data_ptr<float>());
  return c;
}


