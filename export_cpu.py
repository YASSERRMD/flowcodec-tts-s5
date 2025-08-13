import os, argparse, torch
import torch.nn as nn
from prior.flow_matching import PriorFM

def main(prior_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    latent_dim = 384; cond_dim = 512
    model = PriorFM(h=latent_dim, depth=10, heads=6, cond_dim=cond_dim).eval()
    sd = torch.load(prior_path, map_location='cpu'); model.load_state_dict(sd, strict=False)
    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).eval()
    example = (torch.randn(1,300,latent_dim), torch.rand(1,1,latent_dim), torch.randn(1,cond_dim))
    ts = torch.jit.trace(qmodel, example); ts.save(os.path.join(outdir,'prior_q.ts'))
    print('Saved TorchScript:', os.path.join(outdir,'prior_q.ts'))
    try:
        import onnxruntime.quantization as ortq
        onnx_path = os.path.join(outdir,'prior_q.onnx')
        torch.onnx.export(qmodel, example, onnx_path, opset_version=17, input_names=['x_t','t','cond'], output_names=['v_pred'], dynamic_axes={'x_t':{0:'B',1:'T'}, 'cond':{0:'B'}})
        q_path = os.path.join(outdir,'prior_q_int8.onnx')
        ortq.quantize_dynamic(onnx_path, q_path, weight_type=ortq.QuantType.QInt8, optimize_model=True)
        print('Saved ONNX INT8:', q_path)
    except Exception as e:
        print('ONNX export skipped:', e)
if __name__=='__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prior', required=True); ap.add_argument('--outdir', default='cpu_export')
    args = ap.parse_args(); main(args.prior, args.outdir)
