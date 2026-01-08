"""Pathway streaming pipeline for BDH narrative classification."""
import pathway as pw
import torch
import pandas as pd
from pathlib import Path
from model import BDHClassifier
from config import path_config


class BiographySchema(pw.Schema):
    """Schema for biography data."""
    id: int
    book_name: str
    char: str
    caption: str
    content: str


class PredictionSchema(pw.Schema):
    """Schema for predictions."""
    id: int
    prediction: str
    confidence: float
    book_name: str
    char: str


class BDHPathwayClassifier:
    """Pathway-based streaming classifier using BDH."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to trained BDH classifier checkpoint
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading model from {model_path} on {self.device}")
        
        # Load model
        self.model = BDHClassifier.load(model_path, device=self.device)
        self.model.eval()
        
        # Label mapping
        self.label_map = {0: 'consistent', 1: 'contradict'}
    
    def tokenize(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Byte-level tokenization."""
        byte_array = bytearray(text.encode('utf-8'))
        if len(byte_array) > max_length:
            byte_array = byte_array[:max_length]
        tokens = torch.tensor(list(byte_array), dtype=torch.long)
        return tokens
    
    def classify_biography(self, row: pw.Json) -> dict:
        """
        Classify a single biography.
        
        Args:
            row: Pathway row with biography data
        Returns:
            Dictionary with prediction results
        """
        # Extract text
        text = row['content']
        
        # Tokenize
        tokens = self.tokenize(text).unsqueeze(0).to(self.device)  # (1, T)
        attention_mask = torch.ones_like(tokens)
        
        # Predict
        with torch.no_grad():
            predictions, probabilities = self.model.predict(tokens, attention_mask)
        
        # Extract results
        pred_label = predictions[0].item()
        pred_str = self.label_map[pred_label]
        confidence = probabilities[0, pred_label].item()
        
        return {
            'id': row['id'],
            'prediction': pred_str,
            'confidence': float(confidence),
            'book_name': row['book_name'],
            'char': row['char']
        }


def run_streaming_pipeline(
    input_csv: str,
    output_path: str,
    model_path: str,
    device: str = 'cuda'
):
    """
    Run Pathway streaming classification pipeline.
    
    Args:
        input_csv: Path to input CSV (test.csv)
        output_path: Path to output CSV
        model_path: Path to trained model
        device: Device for inference
    """
    # Initialize classifier
    classifier = BDHPathwayClassifier(model_path, device)
    
    # Read CSV as streaming source
    biographies = pw.io.csv.read(
        input_csv,
        schema=BiographySchema,
        mode='static'  # Use 'streaming' for real-time, 'static' for batch
    )
    
    # Apply classification
    predictions = biographies.select(
        id=pw.this.id,
        book_name=pw.this.book_name,
        char=pw.this.char,
        content=pw.this.content
    ).select(
        **pw.apply(classifier.classify_biography, pw.this)
    )
    
    # Write results
    pw.io.csv.write(predictions, output_path)
    
    # Run pipeline
    pw.run()
    
    print(f"Pipeline completed. Results saved to {output_path}")


def run_batch_inference(
    input_csv: str,
    output_csv: str,
    model_path: str,
    device: str = 'cuda',
    batch_size: int = 32
):
    """
    Run batch inference without Pathway (fallback for simplicity).
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to output CSV
        model_path: Path to trained model
        device: Device for inference
        batch_size: Batch size for inference
    """
    print(f"Running batch inference on {input_csv}")
    
    # Load model
    device = device if torch.cuda.is_available() else 'cpu'
    model = BDHClassifier.load(model_path, device=device)
    model.eval()
    
    # Load data
    df = pd.read_csv(input_csv)
    texts = df['content'].tolist()
    
    # Tokenize all texts
    def tokenize(text, max_length=512):
        byte_array = bytearray(text.encode('utf-8'))
        if len(byte_array) > max_length:
            byte_array = byte_array[:max_length]
        return torch.tensor(list(byte_array), dtype=torch.long)
    
    all_predictions = []
    all_confidences = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        batch_tokens = [tokenize(text) for text in batch_texts]
        
        # Pad to same length
        max_len = max(len(t) for t in batch_tokens)
        input_ids = torch.zeros(len(batch_tokens), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_tokens), max_len, dtype=torch.long)
        
        for j, tokens in enumerate(batch_tokens):
            input_ids[j, :len(tokens)] = tokens
            attention_mask[j, :len(tokens)] = 1
        
        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Predict
        with torch.no_grad():
            predictions, probabilities = model.predict(input_ids, attention_mask)
        
        # Store results
        all_predictions.extend(predictions.cpu().tolist())
        
        # Get confidence for predicted class
        for k, pred in enumerate(predictions):
            conf = probabilities[k, pred].item()
            all_confidences.append(conf)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i+len(batch_texts)}/{len(texts)} samples")
    
    # Convert predictions to labels
    label_map = {0: 'consistent', 1: 'contradict'}
    pred_labels = [label_map[p] for p in all_predictions]
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': df['id'],
        'prediction': pred_labels,
        'confidence': all_confidences
    })
    
    # Save
    submission_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # Print statistics
    print(f"\nPrediction distribution:")
    print(submission_df['prediction'].value_counts())
    print(f"\nMean confidence: {submission_df['confidence'].mean():.4f}")
    
    return submission_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BDH Pathway Classifier')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--mode', type=str, default='batch', choices=['pathway', 'batch'],
                        help='Inference mode')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for batch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'pathway':
        run_streaming_pipeline(args.input, args.output, args.model, args.device)
    else:
        run_batch_inference(args.input, args.output, args.model, args.device, args.batch_size)
