"""
Script for fine-tuning a ResNet-50 model on a flower classification dataset.

This script includes:
- Loading image data and labels
- Defining a custom PyTorch Dataset for flower images
- Data transformations for ResNet input
- Training and evaluation loops with progress tracking
- Plotting training and validation metrics (loss, accuracy, F1 score)
- Saving the fine-tuned model

"""

# Import all required libraries
import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Setup device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to data directory and labels file
data_dir = './jpg'
labels_path = './imagelabels.mat'

# Load labels from .mat file
mat = scipy.io.loadmat(labels_path)
labels = mat['labels'][0]

class FlowerDataset(Dataset):
    """
    Custom PyTorch Dataset for loading flower images and their labels.

    Args:
        data_dir (str): Directory containing flower images.
        labels (list or array): List or array of labels corresponding to images.
        transform (callable, optional): Optional transform to be applied on an image.

    Attributes:
        image_files (list): Sorted list of image filenames in data_dir.
    """
    def __init__(self, data_dir, labels, transform=None):
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.image_files = sorted(os.listdir(data_dir))  # Sort images alphabetically

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label) where image is a transformed PIL image tensor,
                   and label is an integer class index (0-based).
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB (3 channels)
        label = self.labels[idx] - 1  # Convert 1-indexed label to 0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations for ResNet input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                         [0.229, 0.224, 0.225])  # Normalize with ImageNet std
])

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    """
    Train the model and evaluate on validation set for a number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (loss function): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        num_epochs (int): Number of epochs to train.

    Tracks and prints training and validation loss, accuracy, and F1 score.
    Plots metrics after training completes.
    """
    # Lists to store metrics for each epoch
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    f1_scores = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training batches
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Clear gradients

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get predicted class indices
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm progress bar with current loss and accuracy
            loop.set_postfix(loss=(running_loss/(total//images.size(0)+1)), acc=100.*correct/total)

        # Calculate training metrics for the epoch
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Evaluate on validation set
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        f1_scores.append(val_f1)

        # Print epoch summary
        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | F1 Score: {val_f1:.4f}')

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, f1_scores)

def evaluate(model, val_loader, criterion):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (loss function): Loss function to compute validation loss.

    Returns:
        tuple: (val_loss, val_accuracy, val_f1) metrics on validation set.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return val_loss, val_accuracy, val_f1

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, f1_scores):
    """
    Plot training and validation loss, accuracy, and F1 score over epochs.

    Args:
        train_losses (list): Training loss values per epoch.
        val_losses (list): Validation loss values per epoch.
        train_accuracies (list): Training accuracy values per epoch.
        val_accuracies (list): Validation accuracy values per epoch.
        f1_scores (list): Validation F1 score values per epoch.
    """
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(15,5))

    # Plot loss comparison
    plt.subplot(1,3,1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy comparison
    plt.subplot(1,3,2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot F1 score
    plt.subplot(1,3,3)
    plt.plot(epochs, f1_scores, label='F1 Score', color='green')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to set up data, model, and start training.

    Steps:
    - Print device being used (CPU or GPU)
    - Load flower dataset and split into training and validation sets
    - Create DataLoader objects for batching
    - Load pre-trained ResNet-50 model and modify final layer for 102 classes
    - Define loss function and optimizer
    - Train the model for specified epochs
    - Save the fine-tuned model to disk
    """
    print(f'Using device: {device}')

    # Load dataset
    dataset = FlowerDataset(data_dir, labels, transform=transform)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)
    num_classes = 102  # Number of flower classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace last fully connected layer
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Start training
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=40)

    # Save the trained model weights
    torch.save(model.state_dict(), 'resnet50_finetuned_flowers2.pth')

# Entry point of the script
if __name__ == '__main__':
    main()
