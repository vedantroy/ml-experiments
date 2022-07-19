from math import ceil
from pathlib import Path
import io
import shutil
import argparse

from supersqlite import sqlite3
from fastargs import Param, Section
from fastargs.decorators import param, get_current_config
from torch.utils.data.dataset import Dataset
import torch
import torchvision
import torchvision.transforms.functional as T

def init_dataset_args():
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
        overwrite=Param(
            bool, "whether to overwrite the output directory if it exists", default=False
        ),
        dataset_dir=Param(
            str,
            "where to write the dataset files",
            default="./data/danbooru/artifacts/shards",
        ),
    )


    Section("logging", "when/where to log").params(
        enabled=Param(bool, "enable logging", default=True),
        logs_dir=Param(
            str,
            "where to log",
            default="./logs",
        ),
        steps=Param(int, "the # of steps between logging images", default=1_000),
    )


TARGET_DIM = 256

def get_tags(conn, img_id):
        tags = conn.execute(
            f'select tag from tags where tags.id = "{img_id}" limit 1'
        ).fetchone()
        if tags == None:
            raise Exception(f"Could not find tag for image: {img_id}")
        tags = tags[0]
        tags = " ".join(sorted(tags.split(" ")))
        return tags

@param("logging.enabled", "log_enabled")
@param("logging.logs_dir", "logs_dir")
@param("logging.steps", "log_steps")
def center_crop_image(path, img_id, idx, log_enabled, logs_dir, log_steps):
        img = torchvision.io.read_image(str(path))
        original = img

        force_log = False

        # apply custom transforms here
        _, H, W = img.shape
        if H > TARGET_DIM and W > TARGET_DIM:
            min_dim = min(H, W)
            scale_down = TARGET_DIM / min_dim
            if scale_down < 0.3:
                # print(f"Found image that was very scaled down, id: {img_id}")
                # force_log = True
                pass
            img = T.resize(img, size=(ceil(H * scale_down), ceil(W * scale_down)))
        img = T.center_crop(img, (TARGET_DIM, TARGET_DIM))

        if log_enabled:
            logs_dir_path = Path(logs_dir)
            assert logs_dir_path.is_dir(), f"Logs dir: {logs_dir} must exist"
            should_log = (idx % log_steps) == 0 or force_log
            if should_log:
                torchvision.io.write_png(
                    original, str(logs_dir_path / (img_id + ".original.png"))
                )
                torchvision.io.write_png(img, str(logs_dir_path / (img_id + ".png")))

        return img

class CenterCropImageCaptionDataset(Dataset):
    def __init__(self, tags_db: str, img_buckets_dir: str):
        tags_db_path = Path(tags_db)
        assert tags_db_path.is_file(), f"No DB file at: {tags_db}"
        self.conn = sqlite3.connect(tags_db)

        imgs_dir_path = Path(img_buckets_dir)
        assert imgs_dir_path.is_dir(), f"No images directory at {img_buckets_dir}"
        self.img_files = list(imgs_dir_path.glob("*/*"))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_id = Path(img_file).stem
        img = center_crop_image(img_file, img_id, idx)
        tags = get_tags(self.conn, img_id)
        return {"img": img, "tags": tags}

def save_tensor(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


@param("files.dataset_dir")
@param("files.overwrite")
def init_dataset_dir(dataset_dir, overwrite):
    dataset_dir_path = Path(dataset_dir)
    dataset_dir_path.mkdir(exist_ok=True)
    try:
        _ = next(dataset_dir_path.iterdir())
        if overwrite:
            dataset_dir_path
            shutil.rmtree(dataset_dir_path)
            dataset_dir_path.mkdir(exist_ok=True)
        else:
            raise Exception(
                f"Dataset dir: {dataset_dir} is non-empty and files.overwrite is false"
            )
    except StopIteration:
        pass

# TODO: Remove default arg
def init_cli_args(description="Construct a streaming dataset"):
    parser = argparse.ArgumentParser(description)
    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode="stderr")
    config.summary()