from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import BrainTumorDataset

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf

def create_dataloaders(base_dir=".", batch_size=32, num_workers=2):
    base_dir = Path(base_dir)
    splits = base_dir / "data" / "splits"

    train_csv = splits / "train.csv"
    val_csv   = splits / "val.csv"
    test_csv  = splits / "test.csv"

    train_tf, eval_tf = get_transforms()

    tmp = BrainTumorDataset(train_csv, transform=None, base_dir=base_dir)
    label_to_index = tmp.label_to_index

    train_ds = BrainTumorDataset(train_csv, transform=train_tf, base_dir=base_dir,
                                 label_to_index=label_to_index)
    val_ds   = BrainTumorDataset(val_csv,   transform=eval_tf,  base_dir=base_dir,
                                 label_to_index=label_to_index)
    test_ds  = BrainTumorDataset(test_csv,  transform=eval_tf,  base_dir=base_dir,
                                 label_to_index=label_to_index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, label_to_index
