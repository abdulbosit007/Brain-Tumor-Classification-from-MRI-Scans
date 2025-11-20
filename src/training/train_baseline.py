import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.models.baseline_cnn import BaselineCNN
from src.data.dataloaders import create_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = output.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, total_correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    train_loader, val_loader, _, label_to_index = create_dataloaders(".")
    num_classes = len(label_to_index)

    model = BaselineCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    EPOCHS = 10

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.3f}")

    # save model
    torch.save(model.state_dict(), "baseline_cnn.pth")
    print("Model saved as baseline_cnn.pth")


if __name__ == "__main__":
    main()
