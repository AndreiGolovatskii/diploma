import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class QrDataset(Dataset):
    QR_CODE_SUFF = 'qr'
    MARKUP_SUFFS = ['rows', 'columns', 'centroids']
    DATA_SUFFS = [QR_CODE_SUFF] + MARKUP_SUFFS

    def __init__(self, data_dir, transform=None, target_transform=None):
        self.img_dir = data_dir
        self.img_uids = list()
        self.transform = transform
        self.target_transform = target_transform

        files = set(filename for filename in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, filename)))
        for filename in files:
            if not filename.endswith(f".{self.QR_CODE_SUFF}.png"):
                continue
            uid = filename.removesuffix(f".{self.QR_CODE_SUFF}.png")
            for suff in self.DATA_SUFFS:
                assert f"{uid}.{suff}.png" in files
            self.img_uids.append(uid)

    def __len__(self):
        return len(self.img_uids)

    def __getitem__(self, idx):
        uid = self.img_uids[idx]

        qr_path = os.path.join(self.img_dir, f"{uid}.{self.QR_CODE_SUFF}.png")
        qr = np.array(Image.open(qr_path).convert('RGB'))

        if self.transform:
            qr = self.transform(qr)

        markups = {}
        for suff in self.MARKUP_SUFFS:
            markup_path = os.path.join(self.img_dir, f"{uid}.{suff}.png")
            markup = np.array(Image.open(markup_path))
            if self.target_transform:
                markup = self.target_transform(markup)
            markups[suff] = markup
        return qr, markups
