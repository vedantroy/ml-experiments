from math import ceil
from pathlib import Path
import subprocess
import pexpect

import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as T
from torch.utils.data.dataset import Dataset


class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir_path: str, caption_server_path: str):
        img_buckets_dir = Path(img_dir_path)
        assert img_buckets_dir.is_dir(), f"Invalid dir: {img_buckets_dir}"
        self.img_files = list(img_buckets_dir.glob("*/*"))
        self.proc = pexpect.spawn(caption_server_path, encoding="utf-8")

    def __len__(self):
        return len(self.img_files)

    def load(self, path):
        path = str(path)
        if (
            (not path.endswith("jpg"))
            and (not path.endswith("jpeg"))
            and (not path.endswith("png"))
        ):
            return None
        img = None
        try:
            img = torchvision.io.read_image(path)
        except Exception:
            return None

        # Image is not RGB
        if img.shape[0] != 3:
            return None

        assert len(img.shape) == 3

        TARGET_DIM = 256
        # resize down by up to 3/10
        C, H, W = img.shape
        if W > TARGET_DIM and H > TARGET_DIM:
            min_dim = min(W, H)
            scale_down = TARGET_DIM / min_dim
            if scale_down < 0.3:
                # This image is so big that scaling it down +
                # center cropping will make it unrecognizable
                return None
            img = T.resize(img, size=(ceil(H * scale_down), ceil(W * scale_down)))
            assert img.shape[1] >= TARGET_DIM
            assert img.shape[2] >= TARGET_DIM

        img = T.center_crop(img, (TARGET_DIM, TARGET_DIM))

        # transform from 0 to 255 => -1 to 1
        img = ((img / 255) * 2) - 1
        # assert torch.max(img).item() <= 1
        # assert torch.min(img).item() >= -1
        return img

    def get_caption(self, img_id):
        self.proc.sendline(img_id)
        self.proc.expect("\n")
        resp = self.proc.before
        if resp.startswith("MISSING_VALUE"):
            return None
        return resp

    def __getitem__(self, idx):
        ordered_tag_str = img = None
        while True:
            idx = (idx + 1) % len(self)
            img_path = self.img_files[idx]
            img = self.load(img_path)
            if img is None:
                continue

            img_id = Path(img_path).stem
            caption = self.get_caption(img_id)
            if not caption:
                continue
            # sort the tags in alphabetical order
            ordered_tag_str = " ".join(sorted(caption.split()))
            break

        return {"img": img, "caption": ordered_tag_str}
