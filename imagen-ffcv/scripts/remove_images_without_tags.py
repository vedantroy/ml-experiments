from pathlib import Path
import argparse
import shutil

import torchvision
import torchvision.transforms.functional as T
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from tqdm import tqdm
from supersqlite import sqlite3

Section("files", "inputs, outputs, etc.").params(
    in_dir=Param(
        str,
        "the source directory with bucket subdirectories",
        default="./data/danbooru/raw/valid_imgs",
    ),
    tags_db=Param(
        str,
        "the sqlite database with id => tag",
        default="./data/danbooru/artifacts/tags.sqlite"
    ),
    allow_delete=Param(
        # fastarg doesn't allow you to set
        # booleans to false in the CLI
        # So, we make the default False
        bool,
        "whether to do a dry run instead of actually deleting files",
        default=False
    )
)

parser = argparse.ArgumentParser(
    description=""
)
config = get_current_config()

config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()


@param("files.in_dir")
@param("files.tags_db")
@param("files.allow_delete")
def run(in_dir, tags_db, allow_delete):
    print(allow_delete)
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
        img_id = path.stem

        tags = conn.execute(
            f'select tag from tags where tags.id = "{img_id}" limit 1'
        ).fetchone()
        if tags == None:
            # print(f"Could not find tag for img: {img_id}")
            deleted += 1
            if allow_delete:
                path.unlink()
            continue
        tags = tags[0]


    print(f"{deleted}/{total} {'would be' if not allow_delete else 'were'} deleted")


run()
