import torch
import torch.nn as nn
from activations import get_activation
from initializers import initialize_weights

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, nun_classes, activation, initializer, sample_input = None):
        super(VGG, self).__init__()
        self.features = self._make_layers(activation, cfg[vgg_name])
        self.classifier = nn.Linear(512, nun_classes)

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, initializer, sample_input))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, activation, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           get_activation(activation=activation)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
def vgg16(num_classes, activation, initializer, sample_input = None):
    return VGG('VGG16', num_classes, activation, initializer, sample_input)

def vgg19(num_classes, activation, initializer, sample_input = None):
    return VGG('VGG19', num_classes, activation, initializer, sample_input)