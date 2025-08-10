import torch
from torch import nn
from kyrie12infer.kernels import rmsnorm as ext_rmsnorm, available as ext_available


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        if ext_available() and x.is_cuda and self.weight.dtype == torch.float32:
            x32 = x.to(torch.float32)
            y32 = ext_rmsnorm(x32, self.weight, self.eps)
            return y32.to(orig_dtype)
        else:
            x32 = x.to(torch.float32)
            var = x32.pow(2).mean(dim=-1, keepdim=True)
            x32.mul_(torch.rsqrt(var + self.eps))
            y = x32.to(orig_dtype).mul_(self.weight)
            return y

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        if ext_available() and x.is_cuda and self.weight.dtype == torch.float32:
            x32 = x.to(torch.float32) + residual.to(torch.float32)
            residual_out = x32.to(orig_dtype)
            y32 = ext_rmsnorm(x32, self.weight, self.eps)
            return y32.to(orig_dtype), residual_out
        else:
            x32 = x.to(torch.float32).add_(residual.to(torch.float32))
            residual_out = x32.to(orig_dtype)
            var = x32.pow(2).mean(dim=-1, keepdim=True)
            x32.mul_(torch.rsqrt(var + self.eps))
            y = x32.to(orig_dtype).mul_(self.weight)
            return y, residual_out

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
