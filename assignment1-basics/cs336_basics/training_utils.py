import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange, einsum
import numpy as np
import os
from typing import BinaryIO, IO
import random
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    if logits.ndim == 3:
        logits_flat = logits.flatten(end_dim=1)
    else:
        logits_flat = logits
    targets_flat = targets.flatten()
    
    labels = logits_flat[torch.arange(logits_flat.size(0)), targets_flat].unsqueeze(-1)
    max_prob = torch.max(logits_flat, dim=-1, keepdim=True)[0]
    
    log_prob = labels - max_prob - torch.log(torch.exp(logits_flat - max_prob).sum(-1))
    
    return -log_prob.mean()

def perplexity(logits: torch.Tensor, targets: torch.Tensor):
    return torch.exp(input=cross_entropy(logits, targets))

def cos_learning_rate_schedule(epoch, warm_up_epochs, cos_annealing_epochs, initial_lr, final_lr):
    if epoch <= warm_up_epochs:
        return epoch / warm_up_epochs * initial_lr
    elif warm_up_epochs < epoch <= cos_annealing_epochs:
        return final_lr + (initial_lr - final_lr) / 2 * (1 + math.cos((epoch - warm_up_epochs) / (cos_annealing_epochs - warm_up_epochs) * math.pi))
    else:
        return final_lr

def gradient_clipping(parameters, max_norm, eps: float = 1e-6):
    params = [p for p in parameters if p.grad is not None]
    device = params[0].grad.device if params else torch.device("cpu")

    # Flatten and compute squared norms
    total_norm_sq = torch.sum(
        torch.stack([
            torch.sum(p.grad.detach().to(dtype=torch.float32).pow(2))
            for p in params
        ])
    )
    total_norm = torch.sqrt(total_norm_sq).to(device)

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)

def data_loading(dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str):
    n = len(dataset)
    max_start = n - context_length - 1

    starts = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.empty((batch_size, context_length), dtype=torch.long, device=device)

    for i, start in enumerate(starts):
        input_slice = np.asarray(dataset[start : start + context_length])
        target_slice = np.asarray(dataset[start + 1 : start + context_length + 1])
        inputs[i] = torch.from_numpy(input_slice)
        targets[i] = torch.from_numpy(target_slice)

    return inputs, targets
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(checkpoint, out)
    

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]