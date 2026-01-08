"""Evaluation utilities for model performance assessment."""
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained BDHClassifier
        dataloader: DataLoader for evaluation
        device: Device to run on
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Predict
            predictions, probabilities = model.predict(input_ids, attention_mask)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    return results


def print_evaluation_report(results):
    """Print comprehensive evaluation report."""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\nAccuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nTrue Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    
    print("\nClassification Report:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['Consistent', 'Contradict']
    ))
    
    print("="*60 + "\n")


def plot_confusion_matrix(cm, output_path='outputs/confusion_matrix.png'):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Consistent', 'Contradict'],
        yticklabels=['Consistent', 'Contradict']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def analyze_errors(results, texts, output_path='outputs/error_analysis.csv'):
    """Analyze misclassified samples."""
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    errors = []
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            errors.append({
                'index': i,
                'text': texts[i][:200] + '...',  # First 200 chars
                'true_label': 'Consistent' if label == 0 else 'Contradict',
                'predicted_label': 'Consistent' if pred == 0 else 'Contradict',
                'confidence': probabilities[i][pred]
            })
    
    if errors:
        error_df = pd.DataFrame(errors)
        error_df.to_csv(output_path, index=False)
        print(f"\nError analysis saved to {output_path}")
        print(f"Total errors: {len(errors)}")
        
        return error_df
    else:
        print("\nNo errors - perfect classification!")
        return None


def generate_submission(
    predictions,
    test_df,
    output_path='outputs/submission.csv'
):
    """
    Generate submission file for hackathon.
    
    Args:
        predictions: Array of predicted labels
        test_df: Original test dataframe
        output_path: Output CSV path
    """
    label_map = {0: 'consistent', 1: 'contradict'}
    pred_labels = [label_map[p] for p in predictions]
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': pred_labels
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission_df


if __name__ == '__main__':
    import argparse
    from model import BDHClassifier
    from data_utils import create_dataloaders
    
    parser = argparse.ArgumentParser(description='Evaluate BDH Classifier')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--train-csv', type=str, default='train.csv', help='Training CSV')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = BDHClassifier.load(args.model, device=args.device)
    
    # Create validation dataloader
    print("Loading validation data...")
    _, val_loader = create_dataloaders(args.train_csv, val_split=0.2)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, val_loader, args.device)
    
    # Print report
    print_evaluation_report(results)
    
    # Save confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plot_confusion_matrix(
        results['confusion_matrix'],
        output_dir / 'confusion_matrix.png'
    )
