import torch
import torch.nn as nn




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
            # Create a sample input with the correct number of channels
            # for the first convolutional layer
            if isinstance(m, nn.Conv2d) and m.in_channels != sample_input.shape[1]:
                sample_input = torch.randn(sample_input.shape[0], m.in_channels, sample_input.shape[2], sample_input.shape[3], device=sample_input.device)
            # Reshape the input for linear layers
            if isinstance(m, nn.Linear):
                sample_input = sample_input.view(sample_input.size(0), -1)  # Flatten the input
                sample_input = sample_input[:, :m.in_features]  # Select the correct number of features

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