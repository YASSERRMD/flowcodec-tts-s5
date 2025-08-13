import torch
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay; self.shadow = {n:p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0-self.decay)
    def apply_to(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])
