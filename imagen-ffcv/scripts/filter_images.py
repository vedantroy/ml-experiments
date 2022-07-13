# I made this script because the ffcv.writer requires
# a Pytorch dataset (with a fixed size, I think?)
# and you can't have a fixed size if you randomly skip corrupt files
# during indexing

# An alternative solution is to assume a fixed size, skip corrupt files
# during indexing, and hope there are enough valid files to match the
# provided length

from pathlib import Path
import argparse
import shutil

import torchvision
import torchvision.transforms.functional as T
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from tqdm.contrib.concurrent import process_map 
from supersqlite import sqlite3

Section("files", "inputs, outputs, etc.").params(
    in_dir=Param(
        str,
        "the directory with bucket subdirectories",
        default="./data/danbooru/raw/valid_imgs",
    ),
    tags_db=Param(
        str,
        "the sqlite database with id => tag",
        default="./data/danbooru/artifacts/tags.sqlite"
    ),
)

parser = argparse.ArgumentParser(
    description="Delete images that cannot be loaded by PIL/torchvision or do not have a tag"
)
config = get_current_config()

config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()


@param("files.in_dir")
@param("files.tags_db")
def run(in_dir, tags_db):
    tags_db_path = Path(tags_db)
    assert tags_db_path.is_file(), f"No DB file at: {tags_db}"
    conn = sqlite3.connect(tags_db)

    in_dir_path = Path(in_dir)
    assert in_dir_path.is_dir(), f"{in_dir} must be a directory"

    img_files = list(Path(in_dir).glob("*/*"))
    total = len(img_files)
    print(f"Processing {total} images")

    deleted = 0
    for path in tqdm(img_files):
        s = str(path)
        if not s.endswith("jpg") and not s.endswith("jpeg") and not s.endswith("png"):
            deleted += 1
            path.unlink()
            continue

        img = None
        try:
            img = torchvision.io.image.read_image(s)
        except Exception:
            path.unlink()
            deleted += 1
            continue

        if img.shape[0] != 3:
            # Not RGB
            path.unlink()
            deleted += 1
            continue
    print(f"Deleted {deleted}/{total}")


run()
