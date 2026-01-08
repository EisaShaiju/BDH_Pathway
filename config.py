"""Configuration for BDH narrative consistency classifier."""
import dataclasses
from pathlib import Path


@dataclasses.dataclass
class BDHConfig:
    """BDH model configuration."""
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256  # Byte-level tokenization
    dropout: float = 0.1
    block_size: int = 512  # Max sequence length


@dataclasses.dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_epochs: int = 50
    patience: int = 10
    gradient_clip: float = 1.0
    val_split: float = 0.2
    device: str = "cuda"  # or "cpu"


@dataclasses.dataclass
class PathConfig:
    """Path configuration."""
    data_dir: Path = Path("data")
    train_csv: Path = Path("train.csv")
    test_csv: Path = Path("test.csv")
    model_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    pretrained_path: Path = Path("models/bdh_pretrained.pt")
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)


# Global configs
bdh_config = BDHConfig()
training_config = TrainingConfig()
path_config = PathConfig()
