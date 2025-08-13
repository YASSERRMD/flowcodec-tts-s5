import torch

def pack_condition(text_enc, speaker_emb, prosody_emb):
    pooled = text_enc.mean(dim=1)
    cond = torch.cat([pooled, speaker_emb, prosody_emb], dim=-1)
    return torch.nn.functional.normalize(cond, dim=-1)
