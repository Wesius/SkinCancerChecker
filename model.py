import os
import csv
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")


class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(csv_path, image_dir):
    images, labels = [], []
    label_map = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading data"):
            img_path = os.path.join(image_dir, row['image'] + '.jpg')
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            try:
                label = [float(row[key]) for key in label_map.keys()]
                label_index = label.index(1.0)
            except ValueError:
                print(f"Warning: Invalid label for image: {row['image']}")
                continue
            images.append(img_path)
            labels.append(label_index)

    print(f"Total images loaded: {len(images)}")
    return images, labels


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SkinLesionModel(nn.Module):
    def __init__(self):
        super(SkinLesionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 26 * 26, 64), nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, criterion, optimizer, scaler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{running_loss / total_predictions:.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })

        print(f'Epoch [{epoch + 1}/{num_epochs}] - '
              f'Loss: {running_loss / total_predictions:.4f}, '
              f'Accuracy: {100 * correct_predictions / total_predictions:.2f}%')


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    label_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    return accuracy


def main():
    # Load and prepare data
    csv_path = 'GroundTruth.csv'
    image_dir = 'images'
    images, labels = load_data(csv_path, image_dir)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)
    print(f"Data split: {len(train_images)} training images, {len(test_images)} test images")

    # Create datasets and dataloaders
    train_dataset = SkinLesionDataset(train_images, train_labels, transform)
    test_dataset = SkinLesionDataset(test_images, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, optimizer, and scaler
    model = SkinLesionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Train the model
    train_model(model, train_loader, criterion, optimizer, scaler, num_epochs=10)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    # Save the model
    model_save_path = 'skin_lesion_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()