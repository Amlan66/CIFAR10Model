import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np
from tqdm import tqdm
from models.modelThree import DilatedNet
from utils.transforms import get_transforms
from torch.optim.lr_scheduler import OneCycleLR
import torchinfo

def train(model, device, train_loader, optimizer, epoch, criterion, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Predict
        pred = model(data)
        
        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update Progress Bar
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} LR={optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_loss/len(train_loader), 100*correct/processed

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def get_model_summary(model, input_size=(1, 3, 32, 32)):
    """
    Print model summary including:
    - Layer details
    - Output shape
    - Number of parameters
    - Estimated total size
    """
    print("\nModel Summary:")
    print("=" * 50)
    summary = torchinfo.summary(model, 
                              input_size=input_size,
                              col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                              depth=4)
    print("\nTotal Parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return summary

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check CUDA availability
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    # Training Parameters
    batch_size = 128
    epochs = 15
    lr = 0.1  # Increased from 0.05 to 0.1
    momentum = 0.9
    
    # Standard CIFAR10 mean and std values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Get transforms
    train_transform, test_transform = get_transforms(mean, std)
    
    # Load CIFAR10 Dataset
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = DilatedNet().to(device)
    get_model_summary(model, input_size=(batch_size, 3, 32, 32))
    
    # Calculate total steps for OneCycleLR
    total_steps = epochs * len(train_loader)
    pct_start = 0.3
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.5,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    criterion = nn.CrossEntropyLoss()
    
    # Replace OneCycleLR with StepLR
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=0.5,
    #     total_steps=total_steps,
    #     pct_start=pct_start,
    #     div_factor=10.0,     # initial_lr = max_lr/div_factor
    #     final_div_factor=1e4,  # min_lr = initial_lr/final_div_factor
    #     three_phase=False,
    #     anneal_strategy='cos'
    # )

    # StepLR configuration
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,  # Step down every 3 epochs
        gamma=0.5     # Reduce learning rate by half
    )
    
    # Training and testing logs
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, criterion, scheduler)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        scheduler.step()  # Step the scheduler here, once per epoch
        
        # Log metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

if __name__ == '__main__':
    main() 