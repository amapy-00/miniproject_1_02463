import torch
import torch.nn as nn

class ParametrizedCNN(nn.Module):
    def __init__(self,
                 conv_channels=[32, 64],
                 fc_layers=[512],
                 dropout=0.5,
                 num_classes=100):
        super(ParametrizedCNN, self).__init__()
        layers = []
        in_channels = 3  # CIFAR100 images have 3 channels
        # Use fixed conv parameters: kernel=3, stride=1, padding=1, pooling kernel=2
        for out_channels in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        fc_input = in_channels
        fc_modules = []
        for fc_dim in fc_layers:
            fc_modules.append(nn.Linear(fc_input, fc_dim))
            fc_modules.append(nn.ReLU(inplace=True))
            fc_modules.append(nn.Dropout(dropout))
            fc_input = fc_dim
        fc_modules.append(nn.Linear(fc_input, num_classes))
        self.classifier = nn.Sequential(*fc_modules)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
