from pathlib import Path
from os import listdir
from os.path import splitext

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch


class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, scale):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, "scale must be between 0 or 1"
        self.scale_img_by = scale

        self.ids = []
        for file_name in listdir(images_dir):
            base, ext = splitext(file_name)
            if not base.startswith("."):
                self.ids.append(base)

        if not self.ids:
            raise RuntimeError(f"No files in {images_dir}")

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"

        # TODO: Figure out what's going on here
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            assert len(img_ndarray.shape) == 3
            # W, H, C => C, W, H
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            # normalize to between 0-1
            # TODO: Why don't we normalize to -1 to 1?
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        return Image.open(filename)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = f"{self.masks_dir}/{name}_mask.gif"

        # img files may have either png or jpg extension
        img_file = list(self.images_dir.glob(name + ".*"))
        assert len(img_file) == 1, f"Either multiple or no images found for id: {name}"

        mask = self.load(mask_file)
        img = self.load(img_file[0])

        assert (
            img.size == mask.size
        ), f"id: {name} had img_size={img.size}, mask_size={mask.size}"
        img = self.preprocess(img, self.scale_img_by, is_mask=False)
        mask = self.preprocess(mask, self.scale_img_by, is_mask=True)

        return dict(
            # TODO: mask is long but and image is float, why?
            # also the contiguous thing
            # nn.conv2D requires a float tensor, the default tensor is a double
            image=torch.as_tensor(img.copy()).float(),
            mask=torch.as_tensor(mask.copy()),
        )
