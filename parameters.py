from dataclasses import dataclass, field
from typing import List, Tuple
import argparse


@dataclass
class Params:
    """
    Configuration dataclass for MLP classification on MNIST.

    Attributes:
        data_dir: Directory to download/load data from.
        num_workers: Number of parallel DataLoader workers.
        mean: Normalization mean for MNIST.
        std: Normalization std for MNIST.
        input_size: Flattened input dimensionality (784 for MNIST).
        hidden_sizes: List of hidden layer widths.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        activation: Activation function ('relu' or 'gelu').
        use_bn: Whether to use BatchNorm1d layers.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per batch.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        l1_lambda: L1 regularization coefficient.
        scheduler: LR scheduler type ('step', 'cosine', or 'none').
        early_stop_patience: Epochs to wait before early stopping.
        seed: Random seed for reproducibility.
        device: Device string ('cpu', 'cuda', or 'mps').
        save_path: File path to save the best model weights.
        log_interval: Batch interval for printing training progress.
        mode: Pipeline mode ('train', 'test', or 'both').
    """

    # Data (fixed for MNIST)
    data_dir:    str              = "./data"
    num_workers: int              = 2
    mean:        Tuple[float, ...] = (0.1307,)
    std:         Tuple[float, ...] = (0.3081,)
    input_size:  int              = 784
    num_classes: int              = 10

    # Model
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout:      float     = 0.3
    activation:   str       = "relu"
    use_bn:       bool      = True

    # Training
    epochs:              int   = 10
    batch_size:          int   = 64
    learning_rate:       float = 1e-3
    weight_decay:        float = 1e-4
    l1_lambda:           float = 0.0
    scheduler:           str   = "step"
    early_stop_patience: int   = 5

    # Misc
    seed:         int = 42
    device:       str = "cpu"
    save_path:    str = "best_model.pth"
    log_interval: int = 100
    mode:         str = "both"


def get_params() -> Params:
    """
    Parse command-line arguments and return a populated Params dataclass.

    Returns:
        Params: A dataclass instance with all configuration values set.

    Example:
        $ python main.py --epochs 20 --lr 1e-3 --hidden_sizes 512 256 128
    """
    parser = argparse.ArgumentParser(description="MLP on MNIST")

    # Model
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128],
                        metavar="H", help="Hidden layer widths e.g. --hidden_sizes 512 256 128")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--use_bn",     type=bool,  default=True)

    # Training
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--weight_decay",         type=float, default=1e-4)
    parser.add_argument("--l1_lambda",            type=float, default=0.0)
    parser.add_argument("--scheduler",  choices=["step", "cosine", "none"], default="step")
    parser.add_argument("--early_stop_patience",  type=int,   default=5)

    # Misc
    parser.add_argument("--device", type=str,                          default="cpu")
    parser.add_argument("--mode",   choices=["train", "test", "both"], default="both")

    args = parser.parse_args()

    return Params(
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        activation=args.activation,
        use_bn=args.use_bn,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        scheduler=args.scheduler,
        early_stop_patience=args.early_stop_patience,
        device=args.device,
        mode=args.mode,
    )