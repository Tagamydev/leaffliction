#%%%

import torch
import tqdm

#%%%
x=torch.rand(5,3)

#print(x)

#%%

torch.cuda.is_available()


#%%

from torch import nn
from torch.utils.data import Dataset, DataLoader


from pathlib import Path
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class CSVDatasetF3(Dataset):
    """
    CSV schema:
        file,f3
        ./images/Apple_scab/image (197).JPG,Apple_scab
        ...

    Returns per item:
        {
          "image": FloatTensor [C,H,W],
          "y":     LongTensor [],   # class index
          "label": str,             # raw label from f3
          "path":  str              # resolved path (for debugging)
        }
    """
    def __init__(self, csv_path, root=".", transforms=None, label_map=None):
        self.root = Path(root)
        self.rows = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append({"file": r["file"], "label": r["f3"]})

        # stable label mapping (sorted unique labels) unless provided
        if label_map is None:
            classes = sorted({r["label"] for r in self.rows})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = dict(label_map)

        # minimal default transforms (224 + imagenet norm); override if you want
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
    


#%%%


ds = CSVDatasetF3("dataset.csv", root=".")
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

# %%

class ImageMLP(nn.Module):
    def __init__(self, in_shape, num_classes, hidden=256, p_drop=0.2):
        super().__init__()
        c,h,w = in_shape
        d = c*h*w
        self.net = nn.Sequential(
            #size 256
            # 1a convolution layer
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
#            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            # 1b convolution layer
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),
            # 2a convolution layer

            # max pooling = 256 / 2 = 128
            nn.MaxPool2d(2, stride=2),
            # 2b convolution layer
            nn.Conv2d(32,64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            # # 2c convolution layer
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # max pooling 128 / 2 = 64
            nn.MaxPool2d(2, stride=2),
            # 3b convolution layer
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            # # 3c convolution layer
            # nn.Conv2d(256, 256, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # 3d convolution layer
            # nn.Conv2d(256, 256, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # max pooling 64 / 2 = 32
            nn.MaxPool2d(2, stride=2),
            # 4b convolution layer
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            # # 4c convolution layer
            # nn.Conv2d(512, 512, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # 4d convolution layer
            # nn.Conv2d(256, 256, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # max pooling 32 / 2 = 16
            nn.MaxPool2d(2, stride=2),

            # 5b convolution layer
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            # # 5c convolution layer
            # nn.Conv2d(512, 512, 3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # 5d convolution layer
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            # max pooling 16 / 2 = 8
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),


            # image size multiplied by chanel number
            # nn.Linear(8 * 8 * 512, 8 * 8 * 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            # nn.Linear(8 * 8 * 512, 8 * 8 * 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p_drop),

            nn.Linear(8 * 8 * 256, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            # image size multiplied by chanel number
            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    ds = CSVDatasetF3("dataset.csv", root=".")
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,      # you can set this to 0 if issues persist
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageMLP(in_shape=(3,256,256), num_classes=len(ds.class_to_idx)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for j in range(10):
        for batch in tqdm.tqdm(loader):   # ← one quick epoch
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("done. last loss:", float(loss))
    # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    #torch.save(model.state_dic(), "./")



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # only needed on Windows
    main()
