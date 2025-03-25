import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
from resnet18 import resnet18
from vgg import vgg16, vgg19
from training import fit, evaluate
import json
import os
from lsuv import lsuv_with_dataloader, lsuv_with_singlebatch

def data_loader_cifar10(batch_size = 256):
    """Returns CIFAR-10 train and test data loaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Download and load the CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    # Download and load the CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1,
        pin_memory=True
    )

    return train_loader, test_loader

def save_results(train_losses, val_losses, train_accs, val_accs, epochs_time, test_loss, test_acc, activation, initializer):
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'BN_cifar10_{activation}_{initializer}.json')
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,  
        'train_accs': train_accs,
        'val_accs': val_accs,
        'epochs_time': epochs_time,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    with open(file_name, 'w') as f:
        json.dump(results, f)

def cifar10(activation, initializer, seed = 0):
    """Run the CIFAR-10 experiment."""
    # Set the seed and device for reproducibility
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Load the CIFAR-10 data
    train_loader, test_loader = data_loader_cifar10()

    # Define the ResNet18 model
    sample_input = None
    if initializer == 'lsuv':
        sample_input = next(iter(train_loader))[0]
    model = vgg19(
        activation=activation, 
        initializer=initializer, 
        sample_input=sample_input, 
        num_classes=10
    ).to(device)

    if initializer == 'lsuv':
        model = lsuv_with_dataloader(model, train_loader, device=device)
    
    # Define the loss function and optimizer
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    num_epochs = 50

    # Train the model
    print(device)
    train_losses, val_losses, train_accs, val_accs, epochs_time = fit(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs
    )

    # Evaluate the model on the test set
    test_loss, test_acc = evaluate(model, criterion, test_loader, device=device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    print(f'Time taken: {epochs_time}')

    # Save to a json file

    save_results(train_losses, val_losses, train_accs, val_accs, epochs_time, test_loss, test_acc, activation, initializer)

if __name__ == "__main__":
    #activations = ['mish','gelu']    
    activations = ['relu', 'elu', 'mish']
    initializers = ['he', 'xavier', 'orthogonal', 'lsuv']
    #initializer = 'lsuv'
    for activation in activations:
        for initializer in initializers:
            cifar10(activation, initializer, seed=0)
    #for activation in activations:
    #cifar10(activation, initializer, seed=0)
