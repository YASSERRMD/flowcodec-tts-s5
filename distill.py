# distill.py
import os
import yaml
import torch
import argparse
from prior.flow_matching import PriorFM

def distill(teacher_path, steps, cfg, out=''):
    """
    Distills a teacher model into a student model.

    Args:
        teacher_path (str): Path to the trained teacher model checkpoint.
        steps (int): Number of distillation steps to perform.
        cfg (dict): Configuration dictionary for the model.
        out (str, optional): Path to save the distilled student model. Defaults to ''.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- FIX 1: Use the CORRECT in_dim from the config (should be 128) ---
    # The teacher was trained on 128-dim latents, so we must build the model
    # with the same architecture to load its weights.
    model_args = {
        'in_dim': cfg['model']['in_dim'],
        'h': cfg['model']['hidden_size'],
        'depth': cfg['model']['num_layers'],
        'heads': cfg['model']['num_heads'],
        'cond_dim': cfg['model']['cond_dim']
    }

    # Initialize teacher and student models with the same, correct arguments
    teacher = PriorFM(**model_args).to(device).eval()
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    print(f"Loaded teacher model from {teacher_path}")

    student = PriorFM(**model_args).to(device).train()
    
    # Initialize the student with the teacher's weights for a good starting point
    student.load_state_dict(teacher.state_dict())
    print("Initialized student model with teacher's weights.")

    # Set up the optimizer for the student model
    opt = torch.optim.AdamW(student.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)

    print(f"Starting distillation for {steps} steps...")
    for it in range(steps):
        # Define dimensions for the synthetic data batch
        B, T = 4, 300
        # --- FIX 2: Generate synthetic data with the correct feature dimension (128) ---
        x1 = torch.randn(B, T, cfg['model']['in_dim'], device=device)
        cond = torch.randn(B, cfg['model']['cond_dim'], device=device)

        # Create 't' as a 1D tensor of shape (B,)
        t = torch.rand(B, device=device)

        # Get the teacher's output (target)
        with torch.no_grad():
            v_t = teacher(x1, t, cond)

        # Get the student's prediction
        v_hat = student(x1, t, cond)

        # Calculate the distillation loss
        loss = ((v_hat - v_t)**2).mean()

        # Perform the optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 20 == 0 or it == steps - 1:
            print(f'Distill iteration [{it+1}/{steps}]: loss = {loss.item():.4f}')

    # Save the final distilled student model
    os.makedirs('checkpoints', exist_ok=True)
    path = out or f'checkpoints/prior_student_{steps}steps.pt'
    torch.save(student.state_dict(), path)
    print(f'Successfully saved distilled student model to {path}')

if __name__ == '__main__':
    # Set up argument parser
    ap = argparse.ArgumentParser(description="Distill a PriorFM teacher model.")
    ap.add_argument('--teacher', required=True, help="Path to the trained teacher model checkpoint.")
    ap.add_argument('--steps', type=int, default=200, help="Number of distillation steps.")
    ap.add_argument('--config', type=str, default='configs/prior_flowmatch.l4_cpu_target.yaml', help="Path to the model configuration YAML file.")
    ap.add_argument('--out', type=str, default='', help="Optional output path for the student model.")
    
    # Parse arguments and run distillation
    args = ap.parse_args()
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    # Ensure the config file has the necessary 'in_dim' key
    if 'in_dim' not in cfg['model']:
        print("Error: Your config file is missing 'model.in_dim'. It should be set to 128.")
        exit(1)

    distill(args.teacher, args.steps, cfg, args.out)
