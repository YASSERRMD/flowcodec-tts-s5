import argparse, os, numpy as np, torch
from datasets import load_dataset, Audio
from tqdm import tqdm
try:
    from encodec import EncodecModel
except Exception as e:
    raise RuntimeError('Please install encodec.') from e

def iter_hf(name, subset, split, sr, streaming, min_sec, max_sec, text_field, speaker_field):
    ds = load_dataset(name, subset or None, split=split, streaming=streaming)
    if not streaming:
        ds = ds.cast_column('audio', Audio(sampling_rate=sr))
    for ex in ds:
        a = ex['audio']; wav = torch.tensor(a['array']).float()
        if wav.ndim>1: wav = wav.mean(-1)
        dur = wav.numel()/sr
        if dur<min_sec or dur>max_sec: continue
        yield {'wav': wav.unsqueeze(0), 'speaker': str(ex.get(speaker_field,'unk')), 'text': ex.get(text_field) or ex.get('text') or ex.get('text_original') or ''}

@torch.inference_mode()
def encode_to_latents(wav, model):
    feats = model.encoder(wav); z = feats[-1]
    z = z.permute(0,2,1).contiguous()
    return z.squeeze(0).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf_name', default='mythicinfinity/libritts_r')
    ap.add_argument('--subset', default='clean')
    ap.add_argument('--split', default='train.clean.100')
    ap.add_argument('--sr', type=int, default=24000)
    ap.add_argument('--bandwidth_kbps', type=float, default=6.0)
    ap.add_argument('--min_sec', type=float, default=1.0)
    ap.add_argument('--max_sec', type=float, default=14.0)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--streaming', action='store_true')
    ap.add_argument('--text_field', default='text_normalized')
    ap.add_argument('--speaker_field', default='speaker_id')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncodecModel.encodec_model_24khz().to(device).eval(); model.set_target_bandwidth(args.bandwidth_kbps)
    it = iter_hf(args.hf_name, args.subset, args.split, args.sr, args.streaming, args.min_sec, args.max_sec, args.text_field, args.speaker_field)
    for i, ex in enumerate(tqdm(it, desc='precompute')):
        wav = ex['wav'].to(device); z = encode_to_latents(wav, model)
        np.savez_compressed(os.path.join(args.out_dir, f'{i:08d}.npz'), z=z, speaker=ex['speaker'], text=ex['text'])
    print('Done. Saved to', args.out_dir)
if __name__=='__main__': main()
