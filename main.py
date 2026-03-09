import random
import ssl

import numpy as np
import torch

from parameters import Params, get_params
from models.MLP import MLP
from train import run_training
from test import run_test


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(params: Params) -> MLP:
    """
    Instantiate the MLP model from params.

    Args:
        params: Configuration dataclass containing model hyperparameters.

    Returns:
        An MLP instance configured according to params.
    """
    return MLP(
        input_size   = params.input_size,
        hidden_sizes = params.hidden_sizes,
        num_classes  = params.num_classes,
        dropout      = params.dropout,
        activation   = params.activation,
        use_bn       = params.use_bn,
    )


def main() -> None:
    """
    Entry point for the MLP MNIST pipeline.

    Parses parameters, sets seed, selects device, builds the model,
    then runs training and/or testing depending on params.mode.
    Run name is auto-generated from key hyperparameters for tracking.
    """
    params = get_params()

    set_seed(params.seed)
    print(f"Seed: {params.seed}")

    device = torch.device(
        params.device if torch.cuda.is_available() else
        "mps"         if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Auto-generate a descriptive run name from key hyperparameters
    hidden_str = "-".join(str(h) for h in params.hidden_sizes)
    run_name = (
        f"hidden={hidden_str}"
        f"_act={params.activation}"
        f"_drop={params.dropout}"
        f"_bn={params.use_bn}"
        f"_wd={params.weight_decay}"
        f"_l1={params.l1_lambda}"
        f"_sched={params.scheduler}"
    )
    print(f"Run: {run_name}")

    model = build_model(params).to(device)
    print(model)

    if params.mode in ("train", "both"):
        run_training(model, params, device, run_name=run_name)

    if params.mode in ("test", "both"):
        run_test(model, params, device, run_name=run_name)


if __name__ == "__main__":
    main()