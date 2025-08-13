# /content/flowcodec-tts-s5/prior/flow_matching.py
import torch, torch.nn as nn
from .dit_blocks import DiTBlock
import math

# A simple helper module for time embeddings
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        args = t[:, None] * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

class PriorFM(nn.Module):
    def __init__(self, in_dim=128, h=384, depth=10, heads=6, cond_dim=768):
        super().__init__()
        self.inp = nn.Linear(in_dim, h)
        self.cproj = nn.Linear(cond_dim, h)
        
        # --- FIX 1: Add a time embedding layer ---
        self.time_embed = nn.Sequential(
            TimestepEmbedding(h),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h)
        )

        # Pass the hidden dim 'h' to the blocks
        self.blocks = nn.ModuleList([DiTBlock(h, heads) for _ in range(depth)])
        self.out = nn.Linear(h, in_dim)
        self.h = h

    def forward(self, x_t, t, cond_vec):
            t_emb = self.time_embed(t) # Shape: (Batch, h)

            # Initial projection
            h = self.inp(x_t) + self.cproj(cond_vec).unsqueeze(1)

            if h.ndim != 3:
                raise ValueError(f"CRITICAL ERROR: 'h' is already {h.ndim}-D before the loop. Something is wrong with the initial projection.")
            
            # Loop through the DiT blocks
            for i, blk in enumerate(self.blocks):
                h = blk(h, t_emb)
                # Add an immediate check to fail fast if the shape is wrong
                if h.ndim != 3:
                    raise ValueError(f"CRITICAL ERROR: Tensor 'h' became {h.ndim}-D after block {i}. Shape is {h.shape}. The DiTBlock is the problem.")
            return self.out(h)
