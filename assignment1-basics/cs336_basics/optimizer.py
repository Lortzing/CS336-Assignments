import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
from collections.abc import Callable, Iterable
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss: float | None = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.01, eps: float = 1e-8, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta_1": betas[0], "beta_2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss: float | None = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            lambda_ = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad.data
                
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0
                    
                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]
   
                # m = beta_1 * m + (1 - beta_1) * grad
                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                self.state[p]["m"] = m
                self.state[p]["v"] = v
                
                state["t"] += 1
                t: int = state["t"]
                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                p.data.mul_(1 - lr * lambda_)
        return loss



if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.