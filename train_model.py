import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# Define the dataset class for HAM10000
class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image'] + '.jpg'  # Add .jpg extension
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None

        image = Image.open(img_path).convert('RGB')

        # Get label as index
        label = self.df.iloc[idx][self.classes].idxmax()
        label_idx = self.class_to_idx[label]

        if self.transform:
            if label_idx == self.class_to_idx['NV']:  # Aggressive augmentations for NV class
                image = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(image)
            else:
                image = self.transform(image)

        return image, label_idx

# Set the path to your CSV file and images directory
csv_file_path = r"C:\Users\harsh\OneDrive\Desktop\HARSHI\HARSHI\SEM 5\AI DEEP LEARNING\project\archive (1)\skin_cancer_data_unique_lastconcat.csv"
img_dir = r"C:\Users\harsh\OneDrive\Desktop\HARSHI\HARSHI\SEM 5\AI DEEP LEARNING\project\archive (1)\images"

df = pd.read_csv(csv_file_path)

# Splitting the dataset into train/validation/test
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']])
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']])

# Define image transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Creating datasets
train_dataset = HAM10000Dataset(train_df, img_dir, transform=transform)
val_dataset = HAM10000Dataset(val_df, img_dir, transform=transform)
test_dataset = HAM10000Dataset(test_df, img_dir, transform=transform)

# Create a weighted sampler for the training data to handle class imbalance
class_counts = train_df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].sum().values
class_weights = 1.0 / class_counts
sample_weights = np.array([class_weights[train_dataset.class_to_idx[train_dataset.df.iloc[i][train_dataset.classes].idxmax()]] for i in range(len(train_dataset))])
weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=weighted_sampler, num_workers=0)  # Set num_workers to 0 to avoid issues on Windows
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha[targets] * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Define the EfficientNet model
def create_efficientnet_model(num_classes, fine_tune=True, pretrained=True):
    model = models.efficientnet_b0(weights='DEFAULT')  # Use weights parameter instead of pretrained
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model

# Initialize the model
num_classes = len(train_dataset.classes)
model = create_efficientnet_model(num_classes, pretrained=True)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function with adjusted class weights (for Focal Loss)
class_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 2.0, 3.0, 4.0]).to(device)  # Increased weight for NV
criterion = FocalLoss(alpha=class_weights)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop with early stopping
patience = 5
best_val_loss = np.inf
epochs_no_improve = 0

num_epochs = 10  # Set to 1 epoch for faster experimentation
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        if images is None:  # Skip if image is None
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'loss': running_loss / len(train_loader),
            'acc': 100. * correct / total
        })

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for images, labels in progress_bar:
            if images is None:  # Skip if image is None
                continue

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'val_loss': running_val_loss / len(val_loader),
                'val_acc': 100. * correct_val / total_val
            })

    val_loss = running_val_loss / len(val_loader)
    val_acc = correct_val / total_val

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        print("Model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping.")
            break
