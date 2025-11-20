import os
import csv
import random
from glob import glob
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
RAW_TRAIN_DIR = BASE_DIR / "data" / "raw" / "train"
RAW_TEST_DIR = BASE_DIR / "data" / "raw" / "test"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

classes = sorted([d.name for d in RAW_TRAIN_DIR.iterdir() if d.is_dir()])
print("Classes:", classes)

def collect_paths(root_dir):
    rows = []
    for cls in classes:
        cls_dir = root_dir / cls
        # support both "No Tumor" or "NoTumor" style names
        if not cls_dir.exists():
            continue
        for img_path in glob(str(cls_dir / "*")):
            rows.append((os.path.relpath(img_path, BASE_DIR), cls))
    return rows

# 1) Collect all train images
train_rows = collect_paths(RAW_TRAIN_DIR)
print(f"Total train images found: {len(train_rows)}")

# 2) Stratified split train -> train + val (80/20)
by_class = {cls: [] for cls in classes}
for path, cls in train_rows:
    by_class[cls].append(path)

final_train = []
final_val = []

for cls, paths in by_class.items():
    random.shuffle(paths)
    n_total = len(paths)
    n_val = int(0.2 * n_total)  # 20% validation
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    final_val += [(p, cls) for p in val_paths]
    final_train += [(p, cls) for p in train_paths]
    print(f"{cls}: train={len(train_paths)}, val={len(val_paths)}")

# 3) Collect test images (kept separate, no split)
test_rows = collect_paths(RAW_TEST_DIR)
print(f"Total test images found: {len(test_rows)}")

def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)

write_csv(final_train, SPLITS_DIR / "train.csv")
write_csv(final_val, SPLITS_DIR / "val.csv")
write_csv(test_rows, SPLITS_DIR / "test.csv")

print("Saved splits to:", SPLITS_DIR)
