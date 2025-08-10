#include <torch/extension.h>

// Declarations of CUDA implementations
at::Tensor reduce_sum_cuda(const at::Tensor& x);
at::Tensor rmsnorm_fwd_cuda(const at::Tensor& input, const at::Tensor& weight, double eps);
at::Tensor sgemm_cuda(const at::Tensor& A, const at::Tensor& B, double alpha, double beta);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum", &reduce_sum_cuda, "Reduce sum over all elements (CUDA)");
  m.def("rmsnorm_fwd", &rmsnorm_fwd_cuda, "Row-wise RMSNorm forward (CUDA)");
  m.def("sgemm", &sgemm_cuda, "Simple SGEMM A(MxK) @ B(KxN) -> C(MxN) (CUDA)");
}


