import torch
from sklearn.metrics import classification_report, confusion_matrix
from datasets import BirdDataset, get_transforms
from models import CLIPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def evaluate_model(predictions, labels, class_names):
    cm = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--weights", required=True, help="Path to the trained model weights (.pth file)")
    args = parser.parse_args()

    # Load configuration
    from cfg import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(cfg.data.image_size)
    val_dataset = BirdDataset(cfg.data.val_dir, transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.train.batch_size)
    
    # Initialize model
    model = CLIPClassifier(num_classes=25, encoder_type=cfg.model.encoder_type).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate metrics and confusion matrix
    evaluate_model(all_preds, all_labels, val_dataset.classes)
