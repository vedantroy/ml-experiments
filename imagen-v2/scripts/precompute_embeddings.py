from pathlib import Path
import argparse
from supersqlite import sqlite3

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME
from tqdm import tqdm

from dataset_writer_utils import save_tensor

Section("files", "inputs, outputs, etc.").params(
    tags_db=Param(
        str,
        "the sqlite database with id => tag",
        default="./data/danbooru/artifacts/tags.sqlite",
    ),
    out_embedding_db=Param(str, "the sqlite database with id => embedding", default="./data/danbooru/artifacts/embeddings.sqlite"),
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
        id, text = self.conn.execute(f"SELECT id, tag FROM tags LIMIT 1 OFFSET {index}").fetchone()
        return { "id": id, "text": text }

@param("files.tags_db")
@param("files.overwrite")
@param("files.out_embedding_db")
def run(tags_db, overwrite, out_embedding_db):
    tags_db_path = Path(tags_db)
    assert tags_db_path.is_file(), f"No DB file at: {tags_db}"

    embedding_db_path = Path(out_embedding_db)
    if embedding_db_path.exists() and not overwrite:
        print(f"{embedding_db_path} already exists, not overwriting")
        return
    embedding_db_path.unlink(missing_ok=True)

    tagsdb_conn = sqlite3.connect(tags_db)
    rows = tagsdb_conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    print(f"Computing embeddings for {rows} texts")


    outdb_conn = sqlite3.connect(out_embedding_db)
    outdb_conn.execute('pragma journal_mode=wal')
    outdb_conn.execute("CREATE TABLE embeddings (id INTEGER, text VARCHAR, embedding BLOB)").fetchall()

    ds = CaptionDataset(tagsdb_conn, rows)
    dl = DataLoader(ds, batch_size=36, shuffle=False, num_workers=1) 

    for batch in tqdm(dl):
        ids, texts = batch["id"], batch["text"]
        embeds, mask = t5_encode_text(
            texts,
            name=DEFAULT_T5_NAME,
            return_attn_mask=True,
        )
        mask = mask.to(torch.int16)
        rows = []
        for row_idx, argmin_idx in enumerate(mask.argmin(dim=1)):
            argmin_idx = argmin_idx.item()
            embed = None
            if argmin_idx != 0:
                embed = embeds[row_idx,:argmin_idx]
            else:
                # This happens if a row is [1, 1, 1]
                # in which case argmin = 1, but
                # that just means this input generated the most tokens
                embed = embeds[row_idx,:]
            embed = embed.cpu()

            embed_bytes = save_tensor(embed)
            rows.append((ids[row_idx].item(), texts[row_idx], embed_bytes))
            
        outdb_conn.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", rows)
    outdb_conn.execute('VACUUM')

run()