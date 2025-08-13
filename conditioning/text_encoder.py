import torch

def text_encode(text_ids):
    B, T = text_ids.shape
    H = 256
    return torch.randn(B, T, H)
