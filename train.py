import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, RESULTS_DIR, PRETRAINED, NUM_CLASSES
)
from data.dataset import get_dataloaders
from models.efficientnet_model import build_efficientnet_b2
from models.densenet_model import build_densenet121
from models.vit_model import build_vit
from models.swin_model import build_swin_t


def get_model(model_name: str) -> nn.Module:
    builders = {
        "efficientnet": build_efficientnet_b2,
        "densenet":     build_densenet121,
        "vit":          build_vit,
        "swin":         build_swin_t,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(builders.keys())}")
    return builders[model_name](num_classes=NUM_CLASSES, pretrained=PRETRAINED)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def run_experiment(model_name: str, magnification: str):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Magnification: {magnification}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    loaders, loss_weights = get_dataloaders(magnification)
    loss_weights = loss_weights.to(device)

    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    ckpt_dir = RESULTS_DIR / model_name / magnification
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_ckpt)
            print(f"    ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved at: {best_ckpt}")
    return history, str(best_ckpt)