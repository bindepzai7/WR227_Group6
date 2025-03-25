import json
import os
import matplotlib.pyplot as plt

activations = ['tanh', 'relu', 'elu', 'gelu', 'silu', 'mish']
act_name = {
    'tanh': 'Tanh',
    'relu': 'ReLU',
    'elu': 'ELU',
    'gelu': 'GELU',
    'silu': 'SiLU',
    'mish': 'Mish'
}

initializers = ['he', 'xavier', 'orthogonal', 'lsuv']
ini_name = {
    'he': 'He',
    'xavier': 'Xavier',
    'orthogonal': 'Orthogonal',
    'lsuv': 'LSUV'
}

# fig, axs = plt.subplots(2, 3, figsize=(20, 9))

# # Store legend handles once
# handles = []
# labels = []

# for i, act in enumerate(activations):
#     row = i // 3
#     col = i % 3
#     ax = axs[row, col]
    
#     for ini in initializers:
#         file_name = f'results/noBN_cifar10_{act}_{ini}.json'
#         if os.path.exists(file_name):
#             with open(file_name, 'r') as f:
#                 results = json.load(f)
#             train_losses = results.get('train_losses', [])
#             if train_losses:
#                 line, = ax.plot(range(1, len(train_losses) + 1), train_losses, label=ini_name[ini])
#                 if i == 0:  # only store handles from first subplot to avoid duplicates
#                     handles.append(line)
#                     labels.append(ini_name[ini])
#         else:
#             print(f"Missing file: {file_name}")
    
#     ax.set_title(act_name[act])
#     ax.set_xlabel('Epoch', labelpad=10)
#     ax.set_ylabel('Training Loss')
#     ax.grid(True)

# # Add one shared legend for all subplots
# fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10)

# # Adjust layout
# # Adjust layout with tight_layout to reduce space
# plt.tight_layout(pad=4.0)
# plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.95)
# plt.savefig('Plots/noBN_train_loss_per_activation.png', dpi=300, bbox_inches='tight')
# plt.show()



for act in activations:
    test_accs = []
    for ini in initializers:
        file_name = f'results/noBN_cifar10_{act}_{ini}.json'
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                results = json.load(f)
            test_acc = results.get('test_acc', 0)
            test_accs.append(test_acc)
        else:
            test_accs.append(0)
    
    # Print in LaTeX format
    print(f'{act_name[act]} & ' + ' & '.join([f'{acc:.3f}' for acc in test_accs]) + ' \\\\')

# import json
# import os
# import matplotlib.pyplot as plt

# epoch_times = []
# labels = []

# for act in activations:
#     file_name = f'results/noBN_cifar10_{act}_xavier.json'
#     if os.path.exists(file_name):
#         with open(file_name, 'r') as f:
#             results = json.load(f)
#         time_val = results.get('epochs_time', None)
#         if isinstance(time_val, (int, float)):
#             epoch_times.append(time_val / 60)  # convert to minutes
#             labels.append(act_name[act])
#         else:
#             print(f"Invalid or missing 'epochs_time' in: {file_name}")
#     else:
#         print(f"Missing file: {file_name}")

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(labels, epoch_times, color='skyblue')

# # Set font size for x-axis labels
# ax.grid(axis='y')

# # Resize the x-axis labels
# plt.xticks(fontsize=20)  # Adjust the font size here
# plt.yticks(fontsize=20)

# plt.tight_layout()

# plt.savefig('Plots/epoch_time_per_activation.png', dpi=300, bbox_inches='tight')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Activation functions and their derivatives
# def tanh(x):
#     return np.tanh(x)

# def tanh_derivative(x):
#     return 1 - np.tanh(x)**2

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# def elu(x, alpha=1.0):
#     return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# def elu_derivative(x, alpha=1.0):
#     return np.where(x >= 0, 1, elu(x, alpha) + alpha)

# def gelu(x):
#     return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# def gelu_derivative(x):
#     # Approximate derivative of GELU
#     tanh_part = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
#     return 0.5 * (1 + tanh_part) + 0.5 * x * (1 - tanh_part**2) * (
#         np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))
    
# def silu(x):
#     return x / (1 + np.exp(-x))

# def silu_derivative(x):
#     sigmoid = 1 / (1 + np.exp(-x))
#     return sigmoid * (1 + x * (1 - sigmoid))

# def mish(x):
#     return x * np.tanh(np.log1p(np.exp(x)))  # x * tanh(softplus(x))

# def mish_derivative(x):
#     sp = np.log1p(np.exp(x))  # softplus
#     tanh_sp = np.tanh(sp)
#     sigmoid = 1 / (1 + np.exp(-x))
#     return tanh_sp + x * sigmoid * (1 - tanh_sp**2)


# x = np.linspace(-6, 6, 400)

# # Setup plot
# fig, axes = plt.subplots(2, 3, figsize=(12, 6))
# axes = axes.flatten()

# functions = [
#     ('Tanh', tanh, tanh_derivative),
#     ('ReLU', relu, relu_derivative),
#     ('ELU', elu, elu_derivative),
#     ('GELU', gelu, gelu_derivative),
#     ('SiLU', silu, silu_derivative),
#     ('Mish', mish, mish_derivative)
# ]

# for i, (name, func, deriv) in enumerate(functions):
#     ax = axes[i]
#     if i == 0:
#         ax.plot(x, func(x), label='Function', linewidth=1)
#         ax.plot(x, deriv(x), label='Derivative', linestyle='--', linewidth=1)
#     else:
#         ax.plot(x, func(x), linewidth=1)
#         ax.plot(x, deriv(x), linestyle='--', linewidth=1)
#     ax.set_title(name)
#     ax.grid(True)

# # Shared legend
# lines, labels = axes[0].get_legend_handles_labels()
# fig.legend(lines, labels, loc='lower center', ncol=2, fontsize=10)

# # Layout and caption
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.savefig('Plots/activation_functions.png', dpi=300, bbox_inches='tight')
# plt.show()