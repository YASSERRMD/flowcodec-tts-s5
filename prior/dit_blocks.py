import torch, torch.nn as nn
class FiLM(nn.Module):
    def __init__(self, h): super().__init__(); self.g=nn.Linear(h,h); self.b=nn.Linear(h,h)
    def forward(self, x, c): return x*(1+self.g(c)) + self.b(c)
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.time_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, t_emb):
        residual_1 = x
        normed_x = self.norm1(x)
        attn_input = normed_x + self.time_proj(t_emb).unsqueeze(1)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = residual_1 + attn_output
        
        residual_2 = x
        mlp_output = self.mlp(self.norm2(x))
        x = residual_2 + mlp_output
        
        return x
