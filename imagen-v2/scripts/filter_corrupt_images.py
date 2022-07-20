# This does not mutate the input images folder
# It ignores bad images on the fly
# It's kind of slow compared to writing shards using a dataloader + mosaic's writer
# (the mutex makes everything twice as slow I think?)
# but ...
# Danbooru is only 5 million images
# This takes ~ 45 mins on 270K images
# For 5 million, that's 13 hours which is OK
# plus, once we start generating our own data, we will know that
# all the images are good, which will let us use the dataloader technique

# TL;DR for future me:
# - The fast way to write data using the Mosaic dataset writer is to:
#   - Use a dataloader with 12+ workers
#   Do the classic for `batch in dl: writer.write_sample(batch)`
# where batch size is 1
# I can't do that with Danbooru since so much of the data is malformed
# So instead, we have this :(
# I'm not sure if this is actually faster than 
# `cp -r imgs valid_imgs && remove_all_bad_images && load_using_dataloader`
# maybe ?? (if I found a CLI program that removed bad images that was faster than
# torchvision.load), but given that I'm only dealing with 5mil images, it's not worth it

from pathlib import Path
import argparse
import shutil
from math import ceil
from threading import Thread, Lock

import torch
import duckdb
import sqlite3
from torch import chunk
import torchvision
import torchvision.transforms.functional as T
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from tqdm.contrib.concurrent import thread_map
from composer.datasets.streaming import StreamingDatasetWriter
from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME

from dataset_writer_utils import save_tensor

Section("files", "inputs, outputs, etc.").params(
    in_dir=Param(
        str,
        "the directory with bucket subdirectories",
        default="./data/danbooru/raw/imgs",
    ),
    out_dir=Param(str, "the directory to output valid files"),
    tags_db=Param(
        str,
        "the sqlite database with id => tag",
        default="./data/danbooru/artifacts/tags.sqlite",
    ),
    overwrite=Param(bool, "delete the destination director", default=False),
)

parser = argparse.ArgumentParser(
    description="Delete images that cannot be loaded by PIL/torchvision"
)
config = get_current_config()

config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()


writer = None
sqlite_conn = None
duckdb_conn = None
deleted = 0
out_dir_path = None
mutex = Lock()


@param("files.in_dir")
@param("files.out_dir")
@param("files.tags_db")
@param("files.overwrite")
def run(in_dir, out_dir, tags_db, overwrite):
    global out_dir_path, deleted, sqlite_conn, duckdb_conn, writer

    tags_db_path = Path(tags_db)
    assert tags_db_path.is_file(), f"No DB file at: {tags_db}"
    sqlite_conn = sqlite3.connect(tags_db, check_same_thread=False)

    in_dir_path = Path(in_dir)
    assert in_dir_path.is_dir(), f"{in_dir} must be a directory"

    if overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    duckdb_path = out_dir_path / "dataset.duckdb"
    assert not duckdb_path.exists(), f"{duckdb_path} already exists"
    duckdb_conn = duckdb.connect(str(duckdb_path), read_only=False)

    duckdb_conn.execute("CREATE TABLE items(id INTEGER, tags VARCHAR[]);")

    writer = StreamingDatasetWriter(
        out_dir_path / "shards",
        ["img", "tags"],
        # ["img", "tags", "embeds"],
        #"masks"],
        shard_size_limit=1 << 24,
    )

    # buckets = list(Path(in_dir).glob("*"))
    # for bucket in buckets:
    #    name = bucket.name
    #    (out_dir_path / name).mkdir(exist_ok=True)

    img_files = list(Path(in_dir).glob("*/*"))
    total = len(img_files)

    # Suprisingly, batching made things slower ...
    chunk_size = 100
    print(f"Processing {total} images in chunks of {chunk_size}")

    def chunker(seq, size):
        return [seq[pos : pos + size] for pos in range(0, len(seq), size)]

    thread_map(process_batch, chunker(img_files, chunk_size), max_workers=12)
    print(f"skipped {deleted}/{total}")

    writer.finish()


def inc_delete():
    global deleted
    deleted += 1


def process_single(path):
    global sqlite_conn

    def clean():
        inc_delete()
        return None

    s = str(path)
    if not s.endswith("jpg") and not s.endswith("jpeg") and not s.endswith("png"):
        return clean()

    # On 230G of files (274552 images)
    # This takes 10-13 mins
    img = None
    try:
        img = torchvision.io.image.read_image(s)
    except Exception:
        return clean()

    if img.shape[0] != 3:
        return clean()

    img_id = None
    try:
        img_id = int(path.stem)
    except Exception:
        return clean()

    # This is basically instant
    tags = sqlite_conn.execute(
        f'select tag from tags where tags.id = "{path.stem}" limit 1'
    ).fetchone()
    if tags == None:
        return clean()

    tags = tags[0]
    tags = sorted(tags.split(" "))

    TARGET_DIM = 256
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

    return img, img_id, tags


def process_batch(paths):
    global out_dir_path, duckdb_conn, writer, mutex

    batch = [process_single(path) for path in paths]
    ids_and_tags = []
    imgs = []
    for item in batch:
        if item is None:
            continue
        img, img_id, tags = item
        imgs.append(img)
        ids_and_tags.append((img_id, tags))

        tags_str = " ".join(tags)

        img_bytes = save_tensor(img)
        #embeds, masks = t5_encode_text(
        #    # function expects an array of strings
        #    [tags_str],
        #    name=DEFAULT_T5_NAME,
        #    return_attn_mask=True,
        #)

        # assert embeds.shape[0] == 1 and masks.shape[0] == 1
        mutex.acquire()
        writer.write_sample({
            "img": img_bytes,
            "tags": tags_str.encode("utf-8"),
            # "embeds": save_tensor(embeds),
            #"masks": save_tensor(masks),
        })
        mutex.release()


    cursor = duckdb_conn.cursor()
    # items = cursor.table("items")
    # for (img_id, tags) in ids_and_tags:
    #    items.insert([img_id, tags])

    # batching does help although tqdm doesn't give a clear indication
    # that this is true (profile w/ scalene to see the difference)
    stmt = "INSERT INTO items VALUES (?, ?)"
    cursor.executemany(stmt, ids_and_tags)

run()
