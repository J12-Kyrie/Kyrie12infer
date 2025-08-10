import os
import torch
from torch.utils.cpp_extension import load

_USE_EXT = os.getenv("KYRIE_USE_EXT", "0") == "1"
_EXT = None

def _maybe_load():
    global _EXT
    if not _USE_EXT:
        return None
    if _EXT is None:
        this_dir = os.path.dirname(__file__)
        sources = [
            os.path.join(this_dir, "ops.cpp"),
            os.path.join(this_dir, "reduce.cu"),
            os.path.join(this_dir, "rmsnorm.cu"),
            os.path.join(this_dir, "sgemm.cu"),
        ]
        _EXT = load(name="kyrie12infer_kernels", sources=sources, verbose=False)
    return _EXT

def available() -> bool:
    return _USE_EXT and torch.cuda.is_available()

def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    ext = _maybe_load()
    if ext is None:
        return x.sum()
    return ext.reduce_sum(x)

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ext = _maybe_load()
    if ext is None:
        # fallback python implementation matching layers.layernorm.RMSNorm
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        y = x32 * torch.rsqrt(var + eps)
        return (y.to(orig_dtype) * weight)
    return ext.rmsnorm_fwd(x, weight, eps)

def sgemm(A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    ext = _maybe_load()
    if ext is None:
        return alpha * (A @ B) + beta * torch.zeros(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
    return ext.sgemm(A, B, alpha, beta)


