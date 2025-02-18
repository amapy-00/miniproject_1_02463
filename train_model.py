import torch
import pytorch_lightning as pl
from model_lightning import ParametrizedCNNLightning
import torchvision
import torchvision.transforms as transforms

def train_model(params, num_epochs=10, n_training_samples=1000):
    # Extract model hyperparameters (keys starting with 'module__')
    model_params = {k.replace('module__', ''): v for k, v in params.items() if k.startswith('module__')}
    # Extract training parameters
    other_params = {k: v for k, v in params.items() if not k.startswith('module__')}
    lr = other_params.get('lr', 1e-3)
    batch_size = other_params.get('batch_size', 32)
    
    # Prepare CIFAR100 data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainset = torch.utils.data.Subset(full_trainset, list(range(n_training_samples)))
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model with hyperparameters and learning rate.
    model = ParametrizedCNNLightning(**model_params, lr=lr)
    
    # Train model using PyTorch Lightning Trainer.
    trainer = pl.Trainer(max_epochs=num_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, trainloader)
    
    # Evaluate on test set.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy

# ...existing code or tests...
