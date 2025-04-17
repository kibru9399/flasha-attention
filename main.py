import torch

a = torch.ones(3, 4, device='cuda')
b = torch.ones(4, 3, device='cuda')
print(a@b, a.device)