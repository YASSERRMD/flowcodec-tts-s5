import torch

def sample_flow_heuristic(prior, cond, frames, latent_dim=384, steps=4):
    x = torch.randn(1, frames, latent_dim)
    ts = torch.linspace(0.8, 0.2, steps)
    for t in ts:
        t_vec = torch.full((1,1,latent_dim), float(t))
        v = prior(x, t_vec, cond)
        x = x + v * (1.0/steps)
    return x
