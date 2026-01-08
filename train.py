"""Training script for BDH narrative consistency classifier."""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

from model import BDHClassifier
from data_utils import create_dataloaders
from config import BDHConfig, TrainingConfig, path_config


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: True class labels
    Returns:
        Dictionary of metrics
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Accuracy
    accuracy = (predictions == labels).mean()
    
    # Per-class metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def train_epoch(
    model: BDHClassifier,
    train_loader,
    optimizer,
    scaler,
    device: str,
    gradient_clip: float = 1.0
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        with autocast():
            logits, loss = model(input_ids, attention_mask, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.append(predictions.detach())
        all_labels.append(labels.detach())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


@torch.no_grad()
def validate(model: BDHClassifier, val_loader, device: str) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits, loss = model(input_ids, attention_mask, labels)
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.append(predictions)
        all_labels.append(labels)
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


def train(
    model: BDHClassifier,
    train_loader,
    val_loader,
    config: TrainingConfig,
    save_path: Path
):
    """
    Full training loop.
    
    Args:
        model: BDH classifier model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        save_path: Path to save best model
    """
    device = config.device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0
    
    for epoch in range(config.max_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, config.gradient_clip
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss or val_metrics['f1'] > best_val_f1:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_f1 = max(val_metrics['f1'], best_val_f1)
            
            model.save(save_path)
            print(f"✓ Saved best model (Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train BDH Classifier')
    parser.add_argument('--train-csv', type=str, default='train.csv', help='Training CSV path')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--pretrained', type=str, default=None, help='Pre-trained BDH checkpoint')
    parser.add_argument('--freeze-bdh', action='store_true', help='Freeze BDH weights')
    parser.add_argument('--output', type=str, default='models/bdh_classifier.pt', help='Output model path')
    
    args = parser.parse_args()
    
    # Update configs
    training_config.batch_size = args.batch_size
    training_config.max_epochs = args.epochs
    training_config.learning_rate = args.lr
    training_config.device = args.device
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        args.train_csv,
        batch_size=training_config.batch_size,
        val_split=training_config.val_split
    )
    
    # Create model
    print("Initializing model...")
    model = BDHClassifier(
        config=bdh_config,
        num_classes=2,
        freeze_bdh=args.freeze_bdh
    )
    
    # Load pre-trained BDH if provided
    if args.pretrained:
        print(f"Loading pre-trained BDH from {args.pretrained}")
        model.load_pretrained_bdh(args.pretrained)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Train
    train(model, train_loader, val_loader, training_config, output_path)


if __name__ == '__main__':
    main()
