from pathlib import Path
import argparse
import sqlite3

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME

Section("files", "inputs, outputs, etc.").params(
    tags_db=Param(
        str,
        "the sqlite database with id => tag",
        default="./data/danbooru/artifacts/tags.sqlite",
    ),
    embedding_db=Param(str, "the sqlite database with id => embedding"),
    overwrite=Param(bool, "delete the destination db", default=False),
)

parser = argparse.ArgumentParser(
    description="compute embeddings for all texts in the db"
)
config = get_current_config()

config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

class CaptionDataset(Dataset):
    def __init__(self, conn, len):
        self.conn = conn
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        text = self.conn.execute(f"SELECT tag FROM tags LIMIT 1 OFFSET {index}").fetchone()
        return text[0]

@param("files.tags_db")
@param("files.embedding_db")
@param("files.overwrite")
def run(tags_db, embedding_db, overwrite):
    tags_db_path = Path(tags_db)
    assert tags_db_path.is_file(), f"No DB file at: {tags_db}"

    embedding_db_path = Path(embedding_db)
    if embedding_db_path.exists() and not overwrite:
        print(f"{embedding_db_path} already exists, not overwriting")
        return
    embedding_db_path.unlink(missing_ok=True)

    sqlite_conn = sqlite3.connect(tags_db, check_same_thread=False)
    rows = sqlite_conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    print(f"Computing embeddings for {rows} texts")

    ds = CaptionDataset(sqlite_conn, rows)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=1) 

    for batch in dl:
        embeds, masks = t5_encode_text(
            # function expects an array of strings
            ["hello world baby", "hai"],
            name=DEFAULT_T5_NAME,
            return_attn_mask=True,
        )
        print(masks)
        #print((masks == 0).nonzero(as_tuple=True))
        raise Exception("e")
        print(embeds.shape)

run()