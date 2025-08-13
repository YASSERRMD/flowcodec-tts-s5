import argparse, torch, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
app = FastAPI()
class TTSIn(BaseModel):
    text: str
    speaker_hint: str | None = None

def load_export(export_dir):
    prior = torch.jit.load(f'{export_dir}/prior_q.ts', map_location='cpu').eval(); return prior

def sample_flow_2step(prior, cond, frames, latent_dim=384):
    x = torch.randn(1, frames, latent_dim)
    for t in (0.66, 0.33):
        t_vec = torch.full((1,1,latent_dim), t); v = prior(x, t_vec, cond); x = x + v*0.5
    return x

@app.post('/tts')
def tts(inp: TTSIn):
    prior = app.state.prior
    text_h = torch.randn(1, 80, 256); spk = torch.randn(1,256); pro = torch.randn(1,256)
    cond = torch.nn.functional.normalize(torch.cat([text_h.mean(1), spk, pro], dim=-1), dim=-1)
    z = sample_flow_2step(prior, cond, frames=240)
    wav = z.squeeze().numpy().astype(np.float32)
    return {'sr':24000, 'samples':wav.tolist()}

if __name__=='__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--export_dir', default='cpu_export'); ap.add_argument('--port', type=int, default=7860)
    args = ap.parse_args(); app.state.prior = load_export(args.export_dir); uvicorn.run(app, host='0.0.0.0', port=args.port)
