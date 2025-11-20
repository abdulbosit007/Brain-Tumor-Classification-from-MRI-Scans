from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, csv_path, transform=None, base_dir=None, label_to_index=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        self.base_dir = Path(base_dir) if base_dir else self.csv_path.parents[2]
        self.transform = transform

        labels = sorted(self.df["label"].unique())
        self.label_to_index = label_to_index or {lbl: i for i, lbl in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row["filepath"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.label_to_index[row["label"]]
        return img, label
