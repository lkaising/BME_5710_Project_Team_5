import torch.nn as nn
from .willnet_se import SEBlock          # reuse your existing SEBlock

class SEResBlock(nn.Module):
    def __init__(self, ch, scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            SEBlock(ch)
        )
        self.scale = scale
    def forward(self, x):
        return x + self.scale * self.body(x)

class WillNetSEDeep(nn.Module):
    """
    9×9 + 5×5 head  →  N (default 8) SE-ResBlocks → 5×5 tail
    Feed **bicubic-upsampled** LR.
    """
    def __init__(self, n_blocks=8, mid_ch=48):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, mid_ch, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[SEResBlock(mid_ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(mid_ch, 1, 5, padding=2)

    def forward(self, x):
        res = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + res
