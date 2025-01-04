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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

# CIFAR10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


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
        scheduler.step()  # Step the scheduler after each batch for OneCycleLR
        
        # Update Progress Bar
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} LR={optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_loss/len(train_loader), 100*correct/processed

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Store misclassified samples
            misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
            if misclassified_mask.any():
                misclassified_images.append(data[misclassified_mask])
                misclassified_labels.append(target[misclassified_mask])
                misclassified_preds.append(pred[misclassified_mask])
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy, misclassified_images, misclassified_labels, misclassified_preds

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

def plot_misclassified(images, labels, preds, classes):
    writer = SummaryWriter('runs/experiment_1')
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.tight_layout()
        plt.imshow(images[i].cpu().squeeze().permute(1, 2, 0))
        plt.title(f'Predicted: {classes[preds[i]]}\nActual: {classes[labels[i]]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    img_grid = torchvision.utils.make_grid(torch.stack(images))
    writer.add_image('Misclassified Images', img_grid)
    writer.close()

def get_misclassified_images(model, test_loader, device):
    model.eval()
    misclassified_images = []
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Find misclassified indices
            misclassified_idx = (pred != target).nonzero(as_tuple=True)[0]
            
            # Store misclassified images and labels
            misclassified_images.extend(data[misclassified_idx])
            true_labels.extend(target[misclassified_idx])
            pred_labels.extend(pred[misclassified_idx])
            
            if len(misclassified_images) >= 10:  # Get at least 10 images
                break
    
    # Convert lists to tensors
    misclassified_images = torch.stack(misclassified_images[:10])
    true_labels = torch.stack(true_labels[:10])
    pred_labels = torch.stack(pred_labels[:10])
    
    return misclassified_images, true_labels, pred_labels

def plot_misclassified(images, true_labels, pred_labels, class_names):
    writer = SummaryWriter('runs/experiment_1')
    
    # If images is already a tensor, don't stack it
    if isinstance(images, torch.Tensor):
        img_grid = torchvision.utils.make_grid(images)
    else:
        # If images is a list of tensors, then stack them
        img_grid = torchvision.utils.make_grid(torch.stack(images))
    
    # Add normalization if needed
    img_grid = img_grid.cpu()  # Move to CPU if it's on GPU
    
    # Log to tensorboard
    writer.add_image('Misclassified Images', img_grid)
    
    # Create figure with class names and predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        img = images[idx].cpu()
        img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
        
        # Denormalize if your images are normalized
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified.png')  # Save the plot as an image file
    plt.close()
    writer.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check CUDA availability
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    # Training Parameters
    batch_size = 128
    epochs = 5
    
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
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001, 
                                weight_decay=1e-4)
    criterion = nn.NLLLoss()
    
    # Calculate total steps for OneCycleLR
    total_steps = epochs * len(train_loader)
    # Add Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.03,              # Peak learning rate
        total_steps=total_steps,
        pct_start=0.3,            # 30% of training in warmup
        div_factor=10,            # Initial LR = max_lr/10
        final_div_factor=300,     # Final LR = max_lr/1000
        anneal_strategy='cos'     # Cosine annealing
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
        test_loss, test_acc, misclassified_images, misclassified_labels, misclassified_preds = test(model, device, test_loader, criterion, epoch)
        
        # Log metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Plot misclassified images from the last epoch
    print("\nDisplaying 10 misclassified images from the last epoch:")
    misclassified_images = torch.cat(misclassified_images)
    misclassified_labels = torch.cat(misclassified_labels)
    misclassified_preds = torch.cat(misclassified_preds)
    plot_misclassified(misclassified_images[:10], 
                      misclassified_labels[:10], 
                      misclassified_preds[:10], 
                      classes)

    # After training is complete
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nGenerating misclassified images...")
    misclassified_images, true_labels, pred_labels = get_misclassified_images(model, test_loader, device)
    plot_misclassified(misclassified_images, true_labels, pred_labels, class_names)
    print("Misclassified images have been saved as 'misclassified.png'")
    print("You can also view them in Tensorboard by running:")
    print("tensorboard --logdir=runs --port=6006")
    print("And then creating an SSH tunnel to your local machine")

if __name__ == '__main__':
    main() 