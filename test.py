import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import your model and dataset classes
from model import SkinLesionModel
from model import SkinLesionDataset, load_data

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_test_data(csv_path, image_dir):
    images, labels = load_data(csv_path, image_dir)
    test_dataset = SkinLesionDataset(images, labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def analyze_predictions(model, test_loader, label_names):
    model.eval()
    all_preds = []
    all_labels = []
    misclassifications = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Analyzing predictions"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Collect misclassifications
            mask = (predicted != labels)
            misclassified = list(zip(labels[mask].cpu().numpy(), predicted[mask].cpu().numpy()))
            misclassifications.extend(misclassified)

    return all_preds, all_labels, misclassifications


def print_misclassification_analysis(misclassifications, label_names):
    misclassification_counts = Counter(misclassifications)
    common_misclassifications = misclassification_counts.most_common()

    print("\nMost Common Misclassifications:")
    for (true_label, pred_label), count in common_misclassifications[:10]:  # Top 10
        print(f"True: {label_names[true_label]}, Predicted: {label_names[pred_label]}, Count: {count}")


def plot_confusion_matrix(all_labels, all_preds, label_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()


def print_classification_report(all_labels, all_preds, label_names):
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))


def main():
    # Load the trained model
    model = SkinLesionModel()
    model.load_state_dict(torch.load('skin_lesion_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    csv_path = 'GroundTruth.csv'  # Adjust this path if needed
    image_dir = 'images'  # Adjust this path if needed
    test_loader = load_test_data(csv_path, image_dir)

    # Define label names
    label_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    # Analyze predictions
    all_preds, all_labels, misclassifications = analyze_predictions(model, test_loader, label_names)

    # Print misclassification analysis
    print_misclassification_analysis(misclassifications, label_names)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, label_names)

    # Print classification report
    print_classification_report(all_labels, all_preds, label_names)


if __name__ == "__main__":
    main()