import torch
import torch.nn.functional as F

_kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
_ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)

def sobel_x(img): return F.conv2d(img, _kx.to(img.device), padding=1)
def sobel_y(img): return F.conv2d(img, _ky.to(img.device), padding=1)
def grad(img): return torch.sqrt(sobel_x(img)**2 + sobel_y(img)**2 + 1e-6)