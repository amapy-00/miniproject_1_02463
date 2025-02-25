import torch
import pytorch_lightning as pl
from model_lightning import ParametrizedCNNLightning
import torchvision
import torchvision.transforms as transforms
import numpy as np

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def train_model(params, num_epochs=10, n_training_samples=1000):
    # If params is a list (from skopt), convert it to a dictionary using the known parameter order.
    if not isinstance(params, dict):
        param_names = [
            'module__base_channels',
            'module__dropout',
        ]
        params = dict(zip(param_names, params))
    seed_everything(sum([115, 107, 105, 98, 105, 100, 105, 32, 116, 111, 105, 108, 101, 116]))
    
    # Extract model hyperparameters.
    model_params = {k.replace('module__', ''): v for k, v in params.items() if k.startswith('module__')}
    # Extract training parameters directly.
    lr = params.get('lr', 1e-3)
    batch_size = int(params.get('batch_size', 32))
    
    # Prepare CIFAR100 data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Balancing train dataset 
    targets = np.array(full_trainset.targets)
    selected_indices = []
    samples_per_class = n_training_samples // params.get("num_classes")
    # Iterate over each class and randomly select the desired number of samples
    for class_idx in range(params.get("num_classes")):
        class_indices = np.where(targets == class_idx)[0]
        selected_class_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        selected_indices.extend(selected_class_indices)
    
    trainset = torch.utils.data.Subset(full_trainset, selected_indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    
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

