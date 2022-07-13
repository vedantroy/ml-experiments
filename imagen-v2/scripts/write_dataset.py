#! /usr/bin/env python3

from math import ceil
from pathlib import Path
import shutil
import argparse
import io

import torch
import torchvision
import torchvision.transforms.functional as T
from fastargs import Param, Section
from fastargs.decorators import param, get_current_config
from tqdm import tqdm
from composer.datasets.streaming import StreamingDatasetWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from supersqlite import sqlite3
from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME, get_encoded_dim

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

        return {"img": img, "tags": tags}


def save_tensor(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()

@param("files.imgs_dir")
@param("files.tags_db")
@param("files.dataset_dir")
@param("files.overwrite")
def run(imgs_dir, tags_db, dataset_dir, overwrite):
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

    ds = ImageCaptionDataset(tags_db=tags_db, img_buckets_dir=imgs_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=12)

    FIELDS = ["img", "tags", "embeddings", "masks", "tokens"]
    with StreamingDatasetWriter(
        dataset_dir, FIELDS, shard_size_limit=1 << 24
    ) as writer:
        for batch in tqdm(dl):
            img, tags = batch["img"], batch["tags"]
            img = img[0]

            # todo: compress more?
            # https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil
            # byte_array = io.BytesIO()
            # T.to_pil_image(img).save(byte_array, format="PNG")

            # IMPORTANT: By training on the raw tensors,
            # you must use the *exact* same Pytorch version
            # that you used to process the images
            # Anything else will cause data drift (I think ???)
            img_bytes = save_tensor(img)

            # TODO: Might not need to return masks:
            embeds, masks = t5_encode_text(
                # function expects an array of strings
                tags, name=DEFAULT_T5_NAME, return_attn_mask=True
            )
            assert embeds.shape[0] == 1 and masks.shape[0] == 1
            n_tokens = embeds.shape[1]
            embeds_bytes = save_tensor(embeds)
            masks_bytes = save_tensor(masks)

            writer.write_sample(
                {
                    "img": img_bytes,
                    # TODO: We might not even need these, but the size
                    # is neglible compared to the size of the embeddings
                    "tags": tags[0].encode("utf-8"),
                    "embeddings": embeds_bytes,
                    "masks": masks_bytes,
                    "tokens": str(n_tokens).encode("utf-8"),
                },
            )


parser = argparse.ArgumentParser(description="Construct a streaming dataset")
config = get_current_config()
config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

run()
