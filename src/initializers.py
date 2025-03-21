import torch
import torch.nn as nn
import math

def delta_orthogonal_(tensor, gain=1.0):
    """
    Applies delta orthogonal initialization to a weight tensor.
    
    For 1D convolution weights (shape: [out_channels, in_channels, kernel_size])
    or 2D convolution weights (shape: [out_channels, in_channels, height, width]),
    this function zeroes out the tensor and assigns an orthogonal matrix to
    the central location (kernel center). The 'gain' factor scales the orthogonal
    matrix.

    Args:
        tensor (torch.Tensor): The weight tensor to initialize.
        gain (float): Scaling factor for the orthogonal matrix.

    Returns:
        torch.Tensor: The initialized tensor (also modified in-place).
        
    Raises:
        ValueError: If the tensor's spatial dimensions are not odd or if its
                    dimensionality is not 3 (conv1d) or 4 (conv2d).
    """
    with torch.no_grad():
        # Conv1d: shape [out_channels, in_channels, kernel_size]
        if tensor.ndim == 3:
            center = tensor.shape[2] // 2
            tensor.zero_()
            # Create an orthogonal matrix of shape [out_channels, in_channels]
            weight = torch.empty(tensor.shape[0], tensor.shape[1], device=tensor.device)
            nn.init.orthogonal_(weight, gain=gain)
            tensor[:, :, center] = weight

        # Conv2d: shape [out_channels, in_channels, height, width]
        elif tensor.ndim == 4:
            # Ensure spatial dimensions are odd (so a unique center exists)
            if tensor.shape[2] % 2 == 0 or tensor.shape[3] % 2 == 0:
                raise ValueError("Delta orthogonal initialization requires odd spatial dimensions.")
            center_h = tensor.shape[2] // 2
            center_w = tensor.shape[3] // 2
            tensor.zero_()
            weight = torch.empty(tensor.shape[0], tensor.shape[1], device=tensor.device)
            nn.init.orthogonal_(weight, gain=gain)
            tensor[:, :, center_h, center_w] = weight

        else:
            raise ValueError("Delta orthogonal initialization only supports 3D (conv1d) or 4D (conv2d) tensors.")
    
    return tensor


def initialize_weights(m, initializer, sample_input=None):
    if initializer == 'he':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif initializer == 'xavier':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif initializer == 'orthogonal':
        if isinstance(m, nn.Conv2d):
            m.weight = delta_orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif initializer == 'lsuv':
        pass
        # if sample_input is None:
        #     raise ValueError("LSUV initialization requires a sample input.")
        # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #     # Initial weight initialization
        #     nn.init.normal_(m.weight, 0, 1)
        #     if m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        #     # Iteratively adjust the weights to achieve unit variance output
        #     tol = 0.1
        #     max_iter = 32
        #     # Create a sample input with the correct number of channels
        #     # for the first convolutional layer
        #     if isinstance(m, nn.Conv2d) and m.in_channels != sample_input.shape[1]:
        #         sample_input = torch.randn(sample_input.shape[0], m.in_channels, sample_input.shape[2], sample_input.shape[3], device=sample_input.device)
        #     # Reshape the input for linear layers
        #     if isinstance(m, nn.Linear):
        #         sample_input = sample_input.view(sample_input.size(0), -1)  # Flatten the input
        #         sample_input = sample_input[:, :m.in_features]  # Select the correct number of features

        #     for i in range(max_iter):
        #         with torch.no_grad():
        #             output = m(sample_input)
        #             variance = output.var().item()
        #             if abs(variance - 1.0) < tol:
        #                 break
        #             scale = 1.0 / (variance**0.5 + 1e-8)
        #             m.weight.data *= scale
    else:
        raise NotImplementedError(f"Initializer {initializer} not implemented")