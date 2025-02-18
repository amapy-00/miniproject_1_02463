import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ParametrizedCNNLightning(pl.LightningModule):
    def __init__(self,
                 conv_channels=[32, 64],
                 fc_layers=[512],
                 dropout=0.5,
                 num_classes=100,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_channels = 3  # CIFAR100 images have 3 channels
        # Fixed conv parameters: kernel=3, stride=1, padding=1, pooling kernel=2
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
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
