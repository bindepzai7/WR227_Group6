import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
from resnet18 import resnet18
from vgg16 import vgg16
from training import fit, evaluate
import json
import os

def data_loader_cifar100(batch_size = 128):
    """Returns CIFAR-10 train and test data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the CIFAR-10 training dataset
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    # Split the dataset into training and validation sets (80% training, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.transform = transform_test

    # Download and load the CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def save_results(train_losses, val_losses, train_accs, val_accs, epochs_time, test_loss, test_acc, activation, initializer):
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'cifar10_{activation}_{initializer}.json')
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

def cifar100(activation, initializer, seed = 0):
    """Run the CIFAR-10 experiment."""
    # Set the seed and device for reproducibility
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the CIFAR-10 data
    train_loader, val_loader, test_loader = data_loader_cifar100()

    # Define the ResNet18 model
    sample_input = None
    if initializer == 'lsuv':
        sample_input = next(iter(train_loader))[0]
    model = resnet18(
        activation=activation, 
        initializer=initializer, 
        sample_input=sample_input, 
        num_classes=100
    ).to(device)
    
    # Define the loss function and optimizer
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    num_epochs = 50

    # Train the model
    print(device)
    train_losses, val_losses, train_accs, val_accs, epochs_time = fit(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs
    )

    # Evaluate the model on the test set
    test_loss, test_acc = evaluate(model, criterion, test_loader, device=device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    # Save to a json file

    save_results(train_losses, val_losses, train_accs, val_accs, epochs_time, test_loss, test_acc, activation, initializer)

if __name__ == "__main__":
    activation = 'gelu'
    initializer = 'he'
    cifar100(activation, initializer, seed=0)
