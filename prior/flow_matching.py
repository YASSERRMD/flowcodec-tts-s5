import torch, torch.nn as nn
from .dit_blocks import DiTBlock
class PriorFM(nn.Module):
    def __init__(self, h=384, depth=10, heads=6, cond_dim=512):
        super().__init__()
        self.inp = nn.Linear(h, h); self.cproj = nn.Linear(cond_dim, h)
        self.blocks = nn.ModuleList([DiTBlock(h, heads) for _ in range(depth)])
        self.out = nn.Linear(h, h); self.h = h
    def forward(self, x_t, t, cond_vec):
        h = self.inp(x_t) + self.cproj(cond_vec).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, cond_vec)
        return self.out(h)
