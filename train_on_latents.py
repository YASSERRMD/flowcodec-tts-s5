import os, json, glob, numpy as np, torch, yaml, argparse, random
from torch.utils.data import Dataset, DataLoader
from utils.ema import EMA
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.bucket_sampler import BucketByLengthSampler
from prior.flow_matching import PriorFM
from prior.conditioner import pack_condition
from conditioning.speaker_ecapa import speaker_embed
from conditioning.prosody_encoder import prosody_embed
from conditioning.text_encoder import text_encode
from losses.fm_loss import flow_matching_loss

class LatentSet(Dataset):
    def __init__(self, latents_dir, index_json=None, max_frames=900):
        self.max_frames = max_frames
        if index_json and os.path.exists(index_json):
            self.index = json.load(open(index_json)); self.paths = [it['path'] for it in self.index]
        else:
            self.index = None; self.paths = sorted(glob.glob(os.path.join(latents_dir, '*.npz')))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i] if isinstance(i, int) else i
        d = np.load(p, allow_pickle=True)
        z = torch.tensor(d['z'], dtype=torch.float32)
        if z.size(0) > self.max_frames:
            start = random.randint(0, z.size(0)-self.max_frames); z = z[start:start+self.max_frames]
        text = str(d['text'].item()) if getattr(d['text'],'ndim',0)==0 else str(d['text'])
        spk = str(d['speaker'].item()) if getattr(d['speaker'],'ndim',0)==0 else str(d['speaker'])
        text_ids = torch.randint(0,60,(min(len(text),120),))
        wav_fake = torch.randn(1, 24000*2)
        return {'z': z, 'text_ids': text_ids, 'wav': wav_fake}

def main(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lat_dir = cfg['latents']['dir']; index_json = cfg['latents'].get('index_json','')
    ds = LatentSet(lat_dir, index_json=index_json, max_frames=cfg['train']['max_frames'])
    if index_json and os.path.exists(index_json):
        sampler = BucketByLengthSampler(index_json, batch_size=1, max_frames=cfg['train']['max_frames'], shuffle=True)
        def batch_iter():
            for paths in sampler:
                for p in paths:
                    yield ds.__getitem__(p)
        iterable = batch_iter()
    else:
        dl = DataLoader(ds, batch_size=1, shuffle=True)

    prior = PriorFM(h=cfg['model']['hidden_size'], depth=cfg['model']['num_layers'], heads=cfg['model']['num_heads'], cond_dim=cfg['model']['cond_dim']).to(device)
    try:
        from bitsandbytes.optim import AdamW8bit as AdamW
    except Exception:
        from torch.optim import AdamW
    optim = AdamW(prior.parameters(), lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']), weight_decay=cfg['train']['weight_decay'])

    ema = EMA(prior, decay=cfg['train']['ema']['decay']) if cfg['train']['ema']['enabled'] else None
    start_step = 0; ckpt_path = 'checkpoints/last.pt'
    if os.path.exists(ckpt_path):
        extra = load_checkpoint(ckpt_path, prior, optim); start_step = extra.get('step',0); print('Resumed from', ckpt_path, 'at step', start_step)

    prior.train(); steps = cfg['train']['steps']; log_int = cfg['train']['log_interval']; ckpt_int = cfg['train']['ckpt_interval']

    def step_batch(b, step):
        z = b['z'].to(device).unsqueeze(0)
        text_h = text_encode(b['text_ids'].unsqueeze(0).to(device)).to(device)
        spk = speaker_embed(b['wav'].to(device)); pro = prosody_embed(b['wav'].to(device))
        cond = pack_condition(text_h, spk, pro)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(cfg['train']['precision']=='bf16')):
            loss = flow_matching_loss(prior, z, cond)
        optim.zero_grad(); loss.backward();
        torch.nn.utils.clip_grad_norm_(prior.parameters(), cfg['train']['grad_clip'])
        optim.step();
        if ema: ema.update(prior)
        if step % log_int == 0: print(f'step {step}: loss={loss.item():.4f}')
        if step % ckpt_int == 0 and step>0:
            extra = {'step': step}
            os.makedirs('checkpoints', exist_ok=True)
            save_checkpoint('checkpoints/last.pt', prior, optim, extra=extra)
            if ema:
                ema.apply_to(prior)
                save_checkpoint('checkpoints/ema.pt', prior, None, extra=extra)

    step = start_step
    if index_json and os.path.exists(index_json):
        for b in iterable:
            step += 1; step_batch(b, step)
            if step >= steps: break
    else:
        for b in dl:
            step += 1; step_batch(b, step)
            if step >= steps: break

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(prior.state_dict(), 'checkpoints/prior_teacher.pt')
    print('Saved checkpoints/prior_teacher.pt')

if __name__=='__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/prior_flowmatch.l4_cpu_target.yaml')
    args = ap.parse_args(); cfg = yaml.safe_load(open(args.config)); main(cfg)
