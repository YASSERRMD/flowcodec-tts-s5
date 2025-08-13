import os, torch

def save_checkpoint(path, model, optimizer=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'model': model.state_dict()}
    if optimizer: state['optim'] = optimizer.state_dict()
    if extra: state['extra'] = extra
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    d = torch.load(path, map_location='cpu')
    model.load_state_dict(d['model'], strict=False)
    if optimizer and 'optim' in d: optimizer.load_state_dict(d['optim'])
    return d.get('extra', {})
