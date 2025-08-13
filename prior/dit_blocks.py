import torch, torch.nn as nn
class FiLM(nn.Module):
    def __init__(self, h): super().__init__(); self.g=nn.Linear(h,h); self.b=nn.Linear(h,h)
    def forward(self, x, c): return x*(1+self.g(c)) + self.b(c)
class DiTBlock(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(h); self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(h); self.ff = nn.Sequential(nn.Linear(h,4*h), nn.GELU(), nn.Linear(4*h,h))
        self.film = FiLM(h)
    def forward(self, x, c):
        y,_ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = self.film(self.norm2(x), c.unsqueeze(1))
        x = x + self.ff(x)
        return x
