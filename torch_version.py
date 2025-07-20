import torch
from torch import nn
from torch.nn import functional as F

print(f"{torch.__version__=}")
print(f"{torch.cuda.is_available()=}")
print(f"{torch.cuda.device_count()=}")