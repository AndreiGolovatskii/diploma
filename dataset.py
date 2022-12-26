import os

from torch.utils.data import Dataset
import numpy as np
import cv2

class QrDataset(Dataset):
    QR_CODE_SUFF = 'qr'
    MARKUP_SUFFS = ['rows', 'columns', 'centroids']
    DATA_SUFFS = [QR_CODE_SUFF] + MARKUP_SUFFS

    def __init__(self, data_dir, transform=None):
        self.img_dir = data_dir
        self.img_uids = list()
        self.transform = transform

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
        qr = cv2.imread(qr_path, cv2.IMREAD_COLOR)

        markups = {}
        for suff in self.MARKUP_SUFFS:
            markup_path = os.path.join(self.img_dir, f"{uid}.{suff}.png")
            markup = np.array(cv2.imread(markup_path, cv2.IMREAD_GRAYSCALE))
            markups[suff] = markup

        if self.transform is not None:
            masks = [markups[key] for key in self.MARKUP_SUFFS]
            transformed = self.transform(image=qr, masks=masks)
            qr = transformed['image']
            markups = {key:transformed['masks'][i] for i, key in enumerate(self.MARKUP_SUFFS)}

        return qr, markups
