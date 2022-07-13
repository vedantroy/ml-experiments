from math import ceil
from pathlib import Path
import argparse

import numpy as np
import torchvision
import torchvision.transforms.functional as T
from fastargs import Param, Section
from fastargs.decorators import param, get_current_config
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, BytesField
from torch.utils.data.dataset import Dataset
from supersqlite import sqlite3

Section("files", "inputs, outputs, etc.").params(
    imgs_dir=Param(
        str,
        "the source directory with bucket subdirectories",
        default="./data/danbooru/raw/valid_imgs",
    ),
    tags_db=Param(
        str,
        "the sqlite database with id => tag",
        default="./data/danbooru/artifacts/tags.sqlite",
    ),
    overwrite_ffcv=Param(
        bool,
        "whether to overwrite the output ffcv file if it exists",
        default=False
    ),
    ffcv_path=Param(
        str,
        "where to write the output ffcv file",
        default="./data/danbooru/artifacts/dataset.beton",
    ),
)

Section("logging", "when/where to log").params(
    enabled=Param(
        bool,
        "enable logging",
        default=True
    ),
    logs_dir=Param(
        str,
        "where to log",
        default="./logs",
    ),
    steps=Param(
        int,
        "the # of steps between logging images",
        default=1_000
    ),
)

TARGET_DIM = 256

class ImageCaptionDataset(Dataset):
    def __init__(self, tags_db: str, img_buckets_dir: str):
        tags_db_path = Path(tags_db)
        assert tags_db_path.is_file(), f"No DB file at: {tags_db}"
        self.conn = sqlite3.connect(tags_db)

        imgs_dir_path = Path(img_buckets_dir)
        assert imgs_dir_path.is_dir(), f"No images directory at {img_buckets_dir}"
        self.img_files = list(imgs_dir_path.glob("*/*"))


    def __len__(self):
        return len(self.img_files)

    @param("logging.enabled", "log_enabled")
    @param("logging.logs_dir", "logs_dir")
    @param("logging.steps", "log_steps")
    def __getitem__(self, idx, log_enabled, logs_dir, log_steps):
        img_file = self.img_files[idx]
        img_id = Path(img_file).stem
        tags = self.conn.execute(
            f'select tag from tags where tags.id = "{img_id}" limit 1'
        ).fetchone()
        if tags == None:
            raise Exception(f"Could not find tag for image: {img_id}")
        tags = tags[0]
        tags = " ".join(sorted(tags.split(" ")))

        img = torchvision.io.read_image(str(img_file))
        original = img

        force_log = False

        # apply custom transforms here
        _, H, W = img.shape
        if H > TARGET_DIM and W > TARGET_DIM:
            min_dim = min(H, W)
            scale_down = TARGET_DIM / min_dim
            if scale_down < 0.3:
                print(f"Found image that was very scaled down, id: {img_id}")
                force_log = True
            img = T.resize(img, size=(ceil(H * scale_down), ceil(W * scale_down)))
        img = T.center_crop(img, (TARGET_DIM, TARGET_DIM))

        if log_enabled:
            logs_dir_path = Path(logs_dir)
            assert logs_dir_path.is_dir(), f"Logs dir: {logs_dir} must exist"
            force_log = False
            should_log = (idx % log_steps) == 0 or force_log
            if should_log:
                torchvision.io.write_png(original, str(logs_dir_path / (img_id + ".original.png")))
                torchvision.io.write_png(img, str(logs_dir_path / (img_id + ".png")))

        tag_buf = np.frombuffer(tags.encode(), dtype=np.uint8)
        return (T.to_pil_image(img), tag_buf)


@param("files.imgs_dir")
@param("files.tags_db")
@param("files.overwrite_ffcv")
@param("files.ffcv_path")
def run(imgs_dir, tags_db, overwrite_ffcv, ffcv_path):
    ffcv_file_ctx = Path(ffcv_path)
    if overwrite_ffcv:
        ffcv_file_ctx.unlink(missing_ok=True)
    assert (
        not ffcv_file_ctx.exists()
    ), f"ffcv file: {ffcv_path} exists, exiting to avoid overwriting"


    writer = DatasetWriter(ffcv_path, {
        # Write the raw bytes (don't use JPEG encoding)
        "img": RGBImageField(write_mode='raw'),
        "tags": BytesField(),
    }, num_workers=1)

    ds = ImageCaptionDataset(tags_db=tags_db, img_buckets_dir=imgs_dir)
    writer.from_indexed_dataset(ds)


parser = argparse.ArgumentParser(
    description="Construct a ffcv dataset"
)
config = get_current_config()
config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

run()
