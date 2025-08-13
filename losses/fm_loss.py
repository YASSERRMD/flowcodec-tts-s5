import torch

def flow_matching_loss(net, x_data, cond):
    """
    Computes the flow matching loss.
    - net: The model predicting the velocity.
    - x_data: The target data tensor, shape (B, T, D).
    - cond: The conditioning tensor.
    """
    # 1. Sample from the noise distribution. Has the same shape as data.
    x0 = torch.randn_like(x_data)

    # 2. Sample the time 't', one value per batch item.
    # Shape is (B, 1, 1) to allow broadcasting over the (T, D) dimensions of x_data.
    t = torch.rand(x_data.size(0), 1, 1, device=x_data.device)

    # 3. Create the interpolated sample x_t. Broadcasting works correctly now.
    x_t = (1 - t) * x0 + t * x_data

    # 4. Define the ground truth velocity vector.
    v_t = x_data - x0

    # 5. Get the model's predicted velocity.
    # Note: The model likely expects 't' as a 1D tensor of shape (B,).
    # Squeezing it removes the singleton dimensions.
    v_hat = net(x_t, t.flatten(), cond)

    # 6. Return the mean squared error between the prediction and the ground truth.
    return torch.mean((v_hat - v_t)**2)
