import os, yaml, torch, argparse
from prior.flow_matching import PriorFM

def distill(teacher_path, steps, cfg, out=''):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher = PriorFM(h=cfg['model']['hidden_size'], depth=cfg['model']['num_layers'], heads=cfg['model']['num_heads'], cond_dim=cfg['model']['cond_dim']).to(device).eval()
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    student = PriorFM(h=cfg['model']['hidden_size'], depth=cfg['model']['num_layers'], heads=cfg['model']['num_heads'], cond_dim=cfg['model']['cond_dim']).to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=2e-4, betas=(0.9,0.95), weight_decay=0.01)
    for it in range(100):
        B,T,H = 1, 300, cfg['model']['hidden_size']
        x1 = torch.randn(B,T,H, device=device)
        cond = torch.randn(B, cfg['model']['cond_dim'], device=device)
        with torch.no_grad():
            t = torch.full((B,1,H), 0.5, device=device); v_t = teacher(x1, t, cond)
        v_hat = student(x1, t, cond)
        loss = ((v_hat - v_t)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 20 == 0: print(f'distill it={it} loss={loss.item():.4f}')
    os.makedirs('checkpoints', exist_ok=True)
    path = out or f'checkpoints/prior_student_{steps}step.pt'
    torch.save(student.state_dict(), path); print('Saved', path)

if __name__=='__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--teacher', required=True); ap.add_argument('--steps', type=int, default=2); ap.add_argument('--config', type=str, default='configs/prior_flowmatch.l4_cpu_target.yaml'); ap.add_argument('--student_out', type=str, default='')
    args = ap.parse_args(); cfg = yaml.safe_load(open(args.config)); distill(args.teacher, args.steps, cfg, args.student_out or '')
