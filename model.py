"""BDH-based classification model for narrative consistency detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from bdh import BDH
from config import BDHConfig


class BDHClassifier(nn.Module):
    """BDH with classification head for binary consistency prediction."""
    
    def __init__(self, config: BDHConfig, num_classes: int = 2, freeze_bdh: bool = False):
        """
        Args:
            config: BDH model configuration
            num_classes: Number of output classes (2 for binary)
            freeze_bdh: Whether to freeze BDH weights during fine-tuning
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Base BDH model
        self.bdh = BDH(config)
        
        # Freeze BDH if pre-trained
        if freeze_bdh:
            for param in self.bdh.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, num_classes)
        )
        
        # Pooling strategy
        self.pool_strategy = "mean"  # Options: "mean", "max", "last", "first"
    
    def pool_embeddings(self, embeddings, attention_mask=None):
        """
        Pool sequence embeddings into fixed-size representation.
        
        Args:
            embeddings: (B, T, n_embd) sequence embeddings
            attention_mask: (B, T) mask for padding tokens
        Returns:
            pooled: (B, n_embd) pooled representation
        """
        if self.pool_strategy == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = embeddings.mean(dim=1)
        
        elif self.pool_strategy == "max":
            pooled = embeddings.max(dim=1)[0]
        
        elif self.pool_strategy == "last":
            if attention_mask is not None:
                # Get last non-padding token
                seq_lengths = attention_mask.sum(dim=1) - 1
                pooled = embeddings[torch.arange(embeddings.size(0)), seq_lengths]
            else:
                pooled = embeddings[:, -1, :]
        
        elif self.pool_strategy == "first":
            pooled = embeddings[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pool_strategy}")
        
        return pooled
    
    def forward(self, idx, attention_mask=None, labels=None):
        """
        Forward pass for classification.
        
        Args:
            idx: (B, T) token indices
            attention_mask: (B, T) attention mask (1 for real tokens, 0 for padding)
            labels: (B,) class labels (optional, for training)
        Returns:
            logits: (B, num_classes) classification logits
            loss: scalar loss (if labels provided)
        """
        # Get BDH embeddings
        embeddings = self.bdh.get_embeddings(idx)  # (B, T, n_embd)
        
        # Pool to fixed size
        pooled = self.pool_embeddings(embeddings, attention_mask)  # (B, n_embd)
        
        # Classification head
        logits = self.classifier(pooled)  # (B, num_classes)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss
    
    def predict(self, idx, attention_mask=None):
        """
        Predict class labels.
        
        Args:
            idx: (B, T) token indices
            attention_mask: (B, T) attention mask
        Returns:
            predictions: (B,) predicted class indices
            probabilities: (B, num_classes) class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(idx, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities
    
    def load_pretrained_bdh(self, checkpoint_path):
        """
        Load pre-trained BDH weights.
        
        Args:
            checkpoint_path: Path to pre-trained BDH checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract BDH state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load into BDH module
        self.bdh.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained BDH from {checkpoint_path}")
    
    def save(self, path):
        """Save full classifier checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'num_classes': self.num_classes,
            'pool_strategy': self.pool_strategy
        }, path)
        print(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load classifier from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model
        model = cls(
            config=checkpoint['config'],
            num_classes=checkpoint['num_classes']
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.pool_strategy = checkpoint.get('pool_strategy', 'mean')
        
        print(f"Loaded model from {path}")
        return model.to(device)
