from pathlib import Path
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class CSVDatasetF3(Dataset):
    """
    Dataset that reads only the 'train' split from a CSV.

    CSV schema:
        path,disease,split
        ./images/Apple_scab/image (197).JPG,Apple_scab,train

    Returns per item:
        {
            "image": FloatTensor [C,H,W],
            "y": LongTensor [],   # class index
            "label": str,         # raw label from disease column
            "path": str           # resolved file path
        }
    """
    def __init__(self, mask, csv_path, root=".", transforms=None, label_map=None):
        self.root = Path(root)
        self.rows = []

        # Read CSV and filter only train split
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"].lower() == mask:
                    self.rows.append({"file": r["path"], "label": r["disease"]})

        if not self.rows:
            raise ValueError("No rows found for the 'train' split.")

        # Create label mapping
        if label_map is None:
            classes = sorted({r["label"] for r in self.rows})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = dict(label_map)

        # Default transforms
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = (self.root / row["file"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transforms(im)
        y = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
        return {"image": x, "y": y, "label": row["label"], "path": str(path)}