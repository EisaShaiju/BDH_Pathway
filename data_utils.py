"""Data utilities for loading and preprocessing narrative datasets."""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import numpy as np


class BiographyDataset(Dataset):
    """Dataset for character biography consistency classification."""
    
    def __init__(self, texts: List[str], labels: List[int] = None, max_length: int = 512):
        """
        Args:
            texts: List of biography text strings
            labels: List of binary labels (0=consistent, 1=contradict)
            max_length: Maximum sequence length (truncate if longer)
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Byte-level tokenization.
        
        Args:
            text: Input text string
        Returns:
            tokens: (T,) tensor of byte values
        """
        # Convert to bytes
        byte_array = bytearray(text.encode('utf-8'))
        
        # Truncate if needed
        if len(byte_array) > self.max_length:
            byte_array = byte_array[:self.max_length]
        
        # Convert to tensor
        tokens = torch.tensor(list(byte_array), dtype=torch.long)
        
        return tokens
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        text = self.texts[idx]
        tokens = self.tokenize(text)
        
        sample = {
            'input_ids': tokens,
            'length': len(tokens)
        }
        
        if self.labels is not None:
            sample['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch with dynamic padding.
    
    Args:
        batch: List of samples from BiographyDataset
    Returns:
        Batched tensors with padding
    """
    # Find max length in batch
    max_len = max(sample['length'] for sample in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    labels = None
    if 'labels' in batch[0]:
        labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, sample in enumerate(batch):
        seq_len = sample['length']
        input_ids[i, :seq_len] = sample['input_ids']
        attention_mask[i, :seq_len] = 1
        
        if labels is not None:
            labels[i] = sample['labels']
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    if labels is not None:
        result['labels'] = labels
    
    return result


def load_data(csv_path: str, max_length: int = 512) -> Tuple[List[str], List[int]]:
    """
    Load data from CSV.
    
    Args:
        csv_path: Path to train.csv or test.csv
        max_length: Maximum sequence length
    Returns:
        texts: List of biography texts
        labels: List of labels (None for test set)
    """
    df = pd.read_csv(csv_path)
    
    # Extract texts
    texts = df['content'].tolist()
    
    # Extract labels if available
    labels = None
    if 'label' in df.columns:
        # Convert "consistent" -> 0, "contradict" -> 1
        label_map = {'consistent': 0, 'contradict': 1}
        labels = [label_map[label] for label in df['label']]
    
    return texts, labels


def create_dataloaders(
    train_csv: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    max_length: int = 512,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_csv: Path to training CSV
        batch_size: Batch size
        val_split: Validation split ratio
        max_length: Maximum sequence length
        random_seed: Random seed for reproducibility
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Load data
    texts, labels = load_data(train_csv, max_length)
    
    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=val_split,
        random_state=random_seed,
        stratify=labels  # Maintain class balance
    )
    
    # Create datasets
    train_dataset = BiographyDataset(train_texts, train_labels, max_length)
    val_dataset = BiographyDataset(val_texts, val_labels, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Class distribution - Train: {np.bincount(train_labels)}, Val: {np.bincount(val_labels)}")
    
    return train_loader, val_loader


def create_test_dataloader(
    test_csv: str,
    batch_size: int = 16,
    max_length: int = 512
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Create test dataloader.
    
    Args:
        test_csv: Path to test CSV
        batch_size: Batch size
        max_length: Maximum sequence length
    Returns:
        test_loader: Test dataloader
        test_df: Original test dataframe for submission generation
    """
    # Load data
    texts, _ = load_data(test_csv, max_length)
    test_df = pd.read_csv(test_csv)
    
    # Create dataset
    test_dataset = BiographyDataset(texts, labels=None, max_length=max_length)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader, test_df
