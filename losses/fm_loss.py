import torch

def flow_matching_loss(net, x_data, cond):
    x0 = torch.randn_like(x_data)
    t  = torch.rand(x_data.size(0), 1, x_data.size(2))
    x_t = (1 - t) * x0 + t * x_data
    v_t = x_data - x0
    v_hat = net(x_t, t, cond)
    return torch.mean((v_hat - v_t)**2)
