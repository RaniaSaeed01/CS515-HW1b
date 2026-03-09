import copy
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import Params


def get_transforms(params: Params) -> transforms.Compose:
    """
    Build the MNIST normalization transform pipeline.

    Args:
        params: Configuration dataclass containing mean and std.

    Returns:
        A composed torchvision transform pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params.mean, params.std),
    ])


def get_loaders(params: Params) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return MNIST training and validation DataLoaders.

    Args:
        params: Configuration dataclass containing data settings.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    tf = get_transforms(params)

    train_ds = datasets.MNIST(params.data_dir, train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST(params.data_dir, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=params.batch_size,
                              shuffle=True,  num_workers=params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=params.batch_size,
                              shuffle=False, num_workers=params.num_workers)
    return train_loader, val_loader


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    params:    Params,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.

    Args:
        model: The MLP model to train.
        loader: DataLoader providing training batches.
        optimizer: Optimizer instance for weight updates.
        criterion: Loss function.
        device: Device to run computations on.
        params: Configuration dataclass containing l1_lambda and log_interval.

    Returns:
        A tuple of (average_loss, accuracy) over the full epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)

        if params.l1_lambda > 0.0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + params.l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % params.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]  "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model: The MLP model to evaluate.
        loader: DataLoader providing validation batches.
        criterion: Loss function.
        device: Device to run computations on.

    Returns:
        A tuple of (average_loss, accuracy) over the full loader.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    params:    Params,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Construct a learning rate scheduler based on params.scheduler.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        params: Configuration dataclass containing scheduler type and epochs.

    Returns:
        An LRScheduler instance, or None if params.scheduler is 'none'.
    """
    if params.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif params.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)
    return None


def run_training(
    model:  nn.Module,
    params: Params,
    device: torch.device,
) -> None:
    """
    Execute the full training loop with validation, checkpointing,
    and early stopping.

    Args:
        model: The MLP model to train.
        params: Configuration dataclass with all training hyperparameters.
        device: Device to run computations on.
    """
    train_loader, val_loader = get_loaders(params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    scheduler = build_scheduler(optimizer, params)

    best_acc         = 0.0
    best_weights     = None
    patience_counter = 0

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")

        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer,
                                            criterion, device, params)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc         = val_acc
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_weights, params.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{params.early_stop_patience}")
            if patience_counter >= params.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")