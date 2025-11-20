print(">>> Script started")

from src.data.dataloaders import create_dataloaders
import torch

def main():
    print(">>> Creating dataloaders...")
    train_loader, val_loader, test_loader, label_to_index = create_dataloaders(".")

    print(">>> Label mapping:", label_to_index)

    x, y = next(iter(train_loader))
    print(">>> Batch images shape:", x.shape)
    print(">>> Batch labels shape:", y.shape)
    print(">>> Unique labels in batch:", torch.unique(y))

if __name__ == "__main__":
    print(">>> Running main()")
    main()
