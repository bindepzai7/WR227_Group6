import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation
from initializers import initialize_weights

# Define the basic residual block used in ResNet18.
class BasicBlock(nn.Module):
    expansion = 1  # Used to compute output channels for blocks.

    def __init__(self, in_channels, out_channels, activation, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
        self.downsample = downsample  # To match dimensions if needed

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If dimensions differ, adjust the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Add the residual
        out = self.act(out)

        return out

# Define the ResNet class
class ResNet18(nn.Module):
    def __init__(self, block, layers, activation, initializer, sample_input = None, num_classes=1000):
        """
        Args:
            block: Block class (BasicBlock for ResNet18)
            layers: List containing the number of blocks in each of the 4 layers
            num_classes: Number of output classes
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer and max pool
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.act = get_activation(activation)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the 4 layers of the network
        self.layer1 = self._make_layer(block, 64,  layers[0], activation, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], activation, stride=2)

        # Average pooling and fully-connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, initializer, sample_input))

    def _make_layer(self, block, out_channels, blocks, activation, stride=1):
        """
        Creates one layer of the ResNet (a sequence of blocks)
        Args:
            block: Block class
            out_channels: Number of channels for this layer
            blocks: Number of blocks to stack
            stride: Stride for the first block in the layer
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Use a 1x1 convolution to match dimensions
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block may need downsampling
        layers.append(block(self.in_channels, out_channels, activation, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Add the remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        #x = self.maxpool(x)

        # Pass through all layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and final classification layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Helper function to instantiate ResNet18
def resnet18(activation, initializer, sample_input = None, num_classes = 1000):
    """Constructs a ResNet-18 model."""
    return ResNet18(BasicBlock, [2, 2, 2, 2], activation, initializer, sample_input, num_classes=num_classes)

# Example usage:
if __name__ == "__main__":
    activation = 'relu'
    initializer = 'he'
    model = resnet18(activation, initializer, num_classes=1000)
    print(model)

    # Test a forward pass with a dummy input tensor
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print("Output shape:", output.shape)
