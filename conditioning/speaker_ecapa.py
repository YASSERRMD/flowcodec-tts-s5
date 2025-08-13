import torch

def speaker_embed(wav):
    return torch.randn(wav.shape[0] if wav.ndim==2 else 1, 256)
