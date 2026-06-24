import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from model import MNISTCNN
import os

def train_model():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Dataset Preprocessing & Loading
    # ToTensor() automatically scales PIL image pixels from [0, 255] to [0.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("Loading MNIST dataset...")
    # Using a local data directory inside the workspace
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # Split training set into Train (90%) and Validation (10%)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 2. Initialize Model, Optimizer, and Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MNISTCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        
        history["train_loss"].append(float(epoch_train_loss))
        history["train_acc"].append(float(epoch_train_acc))
        history["val_loss"].append(float(epoch_val_loss))
        history["val_acc"].append(float(epoch_val_acc))
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc * 100:.2f}%")
              
    # 3. Model Evaluation on Test Set
    print("Evaluating model on test dataset...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("\n--- Test Metrics ---")
    print(f"Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save the model weights
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Saved model weights to mnist_cnn.pth")
    
    # Save training history & test evaluation metrics to JSON
    metrics_summary = {
        "epochs": list(range(1, epochs + 1)),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "test_accuracy": float(test_accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "confusion_matrix": conf_matrix.tolist()
    }
    
    with open("training_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Saved training metrics and evaluation details to training_metrics.json")

if __name__ == "__main__":
    train_model()
