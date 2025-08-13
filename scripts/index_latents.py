import argparse, os, glob, json, numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--latents_dir', required=True); ap.add_argument('--out_json', required=True)
    args = ap.parse_args(); paths = sorted(glob.glob(os.path.join(args.latents_dir, '*.npz')))
    out = []
    for p in paths:
        d = np.load(p); frames = int(d['z'].shape[0]); out.append({'path': p, 'frames': frames})
    with open(args.out_json, 'w') as f: json.dump(out, f)
    print('Wrote index:', args.out_json, 'items:', len(out))
if __name__=='__main__': main()
