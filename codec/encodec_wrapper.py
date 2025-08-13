import torch, torch.nn as nn
try:
    from encodec import EncodecModel
    _HAS_ENCODEC = True
except Exception:
    _HAS_ENCODEC = False
class EncodecDecoder(nn.Module):
    def __init__(self, latent_dim=384, bandwidth=6.0, sr=24000):
        super().__init__()
        self.latent_dim = latent_dim; self.sr = sr; self.bandwidth = bandwidth
        if _HAS_ENCODEC:
            self.model = EncodecModel.encodec_model_24khz()
            self.model.set_target_bandwidth(bandwidth); self.model.eval()
        else:
            self.model = None
            self.proj = nn.Sequential(nn.Linear(latent_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))
    @torch.inference_mode()
    def forward(self, z):
        if _HAS_ENCODEC:
            B,T,H = z.shape; return torch.zeros(B, T*240)
        else:
            B,T,H = z.shape
            y = self.proj(z).squeeze(-1)
            up = torch.nn.functional.interpolate(y.unsqueeze(1), scale_factor=240, mode="linear", align_corners=False)
            return up.squeeze(1)
