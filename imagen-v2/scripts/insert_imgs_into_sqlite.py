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
from threading import Lock
import threading
import time

import duckdb
import sqlite3
import torchvision
import torchvision.transforms.functional as T
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from tqdm.contrib.concurrent import thread_map

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


class TaskRunner:
    @param("files.in_dir")
    @param("files.tags_db")
    @param("files.out_dir")
    @param("files.overwrite")
    def __init__(self, in_dir, tags_db, out_dir, overwrite, use_shards, profile=False):
        in_dir_path = Path(in_dir)
        assert in_dir_path.is_dir(), f"{in_dir} must be a directory"
        self.img_files = list(Path(in_dir).glob("*/*"))
        self.total = len(self.img_files)
        self.skipped = 0
        self.inserted = 0

        tags_db_path = Path(tags_db)
        assert tags_db_path.is_file(), f"No tags DB file at: {tags_db}"
        self.tags_db_conn = sqlite3.connect(tags_db, check_same_thread=False)

        if overwrite:
            shutil.rmtree(out_dir, ignore_errors=True)

        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        duckdb_path = out_dir_path / "dataset.duckdb"
        assert not duckdb_path.exists(), f"{duckdb_path} already exists"
        duckdb_conn = duckdb.connect(str(duckdb_path), read_only=False)
        duckdb_conn.execute(
            "CREATE TABLE items(id INTEGER, tags VARCHAR[], PRIMARY KEY(id));"
        )
        self.duckdb_conn = duckdb_conn

        self.use_shards = use_shards
        self.shards_path = out_dir_path / "shards"
        self.shards_path.mkdir()
        self.sqlite_pool = {}
        self.locks = {}
        self.profile = profile

    def make_sqlite_conn(self, path):
            # when using shards, each connection gets its own DB
            # without shards, we do manual locking
            conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            conn.execute("pragma journal_mode=wal")
            conn.execute("CREATE TABLE imgs(id INTEGER, tags VARCHAR, img BLOB, PRIMARY KEY(id));")
            return conn

    def get_current_milli_time(self):
        return round(time.time() * 1000)

    def acquire_lock(self, lock):
            before = self.get_current_milli_time()
            lock.acquire()
            after = self.get_current_milli_time()
            if self.profile:
                print(f"Acquired lock in {after - before}ms")

    def get_tid(self):
        if self.use_shards:
            return threading.get_native_id()
        else:
            return 0

    def acquire_sqlite_conn(self):
        tid = self.get_tid()
        if tid not in self.sqlite_pool:
            conn = self.make_sqlite_conn(self.shards_path / f"shard-{tid}.sqlite")
            self.sqlite_pool[tid] = conn
            self.locks[tid] = Lock()
        self.acquire_lock(self.locks[tid])
        return self.sqlite_pool[tid]

    def release_sqlite_conn(self):
        tid = self.get_tid()
        self.locks[tid].release()

    def cleanup(self):
        print("STARTING CLEANUP ...")
        for (tid, conn) in self.sqlite_pool.items():
            lock = self.locks[tid]
            print(f"Waiting to acquire lock: {tid}")
            lock.acquire()
            conn.execute("VACUUM;")
            conn.close()
            lock.release()

    def run(self):
        chunk_size = 100
        print(f"Processing {self.total} images in chunks of {chunk_size}")

        def chunker(seq, size):
            return [seq[pos : pos + size] for pos in range(0, len(seq), size)]

        try:
            thread_map(
                self.process_batch, chunker(self.img_files, chunk_size), max_workers=12
            )
        except KeyboardInterrupt as e:
            print("CAUGHT KEYBOARD INTERRUPT, EXITING...")
            self.cleanup()
            self.duckdb_conn.close()
        print(f"skipped {self.skipped}/{self.inserted}")

    def process_single(self, path):
        def clean():
            self.skipped += 1
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
        tags = self.tags_db_conn.execute(
            f'select tag from tags where tags.id = "{path.stem}" limit 1'
        ).fetchone()
        if tags == None:
            return clean()

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

    def process_batch(self, paths):
        batch_start = self.get_current_milli_time()
        batch = [self.process_single(path) for path in paths]
        ids_and_tag_arrays = []
        ids_and_imgs_and_tags = []

        for item in batch:
            if item is None:
                continue
            self.inserted += 1
            img, img_id, tags = item

            tags = tags[0]
            tags = sorted(tags.split(" "))
            ids_and_tag_arrays.append((img_id, tags))

            tags_str = " ".join(tags)
            img_bytes = save_tensor(img)
            ids_and_imgs_and_tags.append((img_id, img_bytes, tags_str))
        batch_end = self.get_current_milli_time()

        insert_imgs_start = self.get_current_milli_time()
        conn = self.acquire_sqlite_conn()
        conn.executemany("INSERT INTO imgs VALUES (?, ?, ?)", ids_and_imgs_and_tags)
        self.release_sqlite_conn()
        insert_imgs_end = self.get_current_milli_time()

        insert_metadata_start = self.get_current_milli_time()
        cursor = self.duckdb_conn.cursor()
        # batching does help although tqdm doesn't give a clear indication
        # that this is true (profile w/ scalene to see the difference)
        cursor.executemany("INSERT INTO items VALUES (?, ?)", ids_and_tag_arrays)
        insert_metadata_end = self.get_current_milli_time()

        if self.profile:
            batch_time = batch_end - batch_start
            insert_imgs_time = insert_imgs_end - insert_imgs_start
            insert_metadata_time = insert_metadata_end - insert_metadata_start
            total_time = batch_time + insert_imgs_time + insert_metadata_time

            batch_time_percent = batch_time / total_time * 100
            insert_imgs_time_percent = insert_imgs_time / total_time * 100
            insert_metadata_time_percent = insert_metadata_time / total_time * 100

            print("Batch time:", batch_time, "ms", f"({batch_time_percent:.2f}%)")
            print("Insert imgs time:", insert_imgs_time, "ms", f"({insert_imgs_time_percent:.2f}%)")
            print("Insert metadata time:", insert_metadata_time, "ms", f"({insert_metadata_time_percent:.2f}%)")

# With sharding:
# - Batch takes 50-70% of time
# - Inserting imgs takes ~10-20% of time
# - Inserting metadata takes ~10% of time
# 400K images takes 50mins - 1hr w/o shards
# I don't notice a huge difference w/ or w/o sharding?

# Sharding does help (I think)
# w/o sharding
# - Inserting imgs takes ~15-30% of time
# Lock acquisition times can be up to 5 seconds

# 365G img data => ~ 85G img data (due to cropping / etc)
runner = TaskRunner(use_shards=True, profile=False)
runner.run()