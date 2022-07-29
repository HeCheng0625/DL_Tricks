import torch
import torch.nn as nn

class FGM():
    """
    example:
        fgm = FGM(model)
        for batch_input, batch_label in data:
            optimizer.grad()
            batch_output = model(batch_input)
            loss = loss_fn(batch_output, batch_label)
            loss.backward()

            fgm.attack()
            loss_adv = model(model(batch_input), batch_label)
            loss_adv.backward()
            fgm.restore()

            optimizer.step()
    """
    def __init__(self, model, emb_name='emb.'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}

    def attack(self, epsilon=1):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.data / norm
                    param.data.add_(r_at)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}