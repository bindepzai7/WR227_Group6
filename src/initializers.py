import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m, initializer, sample_input=None):
    if initializer == 'he':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif initializer == 'xavier':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif initializer == 'orthogonal':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif initializer == 'lsuv':
        if sample_input is None:
            raise ValueError("LSUV initialization requires a sample input.")
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # Initial weight initialization
            nn.init.normal_(m.weight, 0, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # Iteratively adjust the weights to achieve unit variance output
            tol = 0.1
            max_iter = 20
            for i in range(max_iter):
                with torch.no_grad():
                    output = m(sample_input)
                    variance = output.var().item()
                    if abs(variance - 1.0) < tol:
                        break
                    scale = 1.0 / (variance**0.5 + 1e-8)
                    m.weight.data *= scale
    else:
        raise NotImplementedError(f"Initializer {initializer} not implemented")
