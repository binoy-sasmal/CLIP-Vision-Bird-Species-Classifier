import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import BirdDataset, get_transforms
from models import CLIPClassifier
from tqdm import tqdm  # Import tqdm for progress bar
from sklearn.metrics import accuracy_score
import argparse

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    from cfg import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    # Prepare dataset and dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(cfg.data.image_size)
    train_dataset = BirdDataset(cfg.data.train_dir, transform)
    val_dataset = BirdDataset(cfg.data.val_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size)

    # Model and optimizer
    model = CLIPClassifier(num_classes=25, encoder_type=cfg.model.encoder_type).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(cfg.train.epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        # Use tqdm to wrap the DataLoader for progress tracking
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}, Average Loss: {total_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Save the trained model weights
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "clip_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
