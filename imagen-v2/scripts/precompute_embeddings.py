from pathlib import Path
import argparse
import duckdb
import sqlite3

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from fastargs import Param, Section 
from fastargs.decorators import param
from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME
from tqdm import tqdm

from utils import save_tensor, init_cli_args, get_current_milli_time

assert torch.cuda.is_available()

Section("files", "inputs, outputs, etc.").params(
    tags_db=Param(
        str,
        "the duckdb database with id => tag",
        default="./dataset/dataset.duckdb",
    ),
    out_embedding_db=Param(str, "the sqlite database with id => embedding", default="./dataset/embeddings.sqlite"),
    overwrite=Param(bool, "delete the destination db", default=False),
)

Section("data", "how to feed data to the model").params(
    batch_size=Param(int, "the batch size", default=32),
    workers=Param(int, "the # of workers", default=4)
)

init_cli_args("Precompute embeddings")

class CaptionDataset(Dataset):
    def __init__(self, conn, len, profile=True):
        self.conn = conn.cursor()
        self.len = len
        self.profile = profile

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # CAUTION: I don't sort the text b/c I already sorted it before inserting it into DuckDB
        # Make sure this continues to be the case!

        start = get_current_milli_time()
        id, text = self.conn.execute(f"SELECT id, tags as text FROM items LIMIT 1 OFFSET {index}").fetchone()
        text = " ".join(text)
        # print(text)
        end = get_current_milli_time()

        if self.profile:
            print(f"query took {end - start}ms")

        # hack until I fix: https://github.com/duckdb/duckdb/issues/4169 (go to bottom)
        #text = text.replace(",", " ")[1:-1]
        return { "id": id, "text": text }

def process_batches(conn, dl):
    profile = False

    for batch in tqdm(dl):
        if profile:
            print("Starting batch")
        ids, texts = batch["id"], batch["text"]
        start = get_current_milli_time()
        # if stuff crashes with an opaque error, it's probably b/c
        # your computer doesn't have enough memory -- run htop
        # using the t5-xl requires ~ 20 gigs
        embeds, mask = t5_encode_text(
            texts,
            #name=DEFAULT_T5_NAME,
            name="google/t5-v1_1-xl",
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
        end = get_current_milli_time()
        if profile:
            print(f"generating embeddings: {end - start}ms")
        conn.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", rows)

@param("files.tags_db")
@param("files.overwrite")
@param("files.out_embedding_db")
@param("data.batch_size")
@param("data.workers")
def run(tags_db, overwrite, out_embedding_db, batch_size, workers):
    tags_db_path = Path(tags_db)
    assert tags_db_path.is_file(), f"No DB file at: {tags_db}"

    embedding_db_path = Path(out_embedding_db)
    if embedding_db_path.exists() and not overwrite:
        print(f"{embedding_db_path} already exists, not overwriting")
        return
    embedding_db_path.unlink(missing_ok=True)

    conn = duckdb.connect(tags_db, read_only=True)
    rows = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"Computing embeddings for {rows} texts")

    # https://github.com/ghaering/pysqlite/issues/109
    # Prevent sqlite3 from implicitly starting a transaction
    outdb_conn = sqlite3.connect(out_embedding_db, isolation_level=None).cursor()
    outdb_conn.execute('pragma journal_mode=wal')
    outdb_conn.execute("CREATE TABLE embeddings (id INTEGER, text VARCHAR, embedding BLOB, PRIMARY KEY (id))").fetchall()

    ds = CaptionDataset(conn, len=rows, profile=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    try:
        process_batches(outdb_conn, dl)
    except KeyboardInterrupt as e:
        print("Interrupted, saving DB ...")
        outdb_conn.execute('VACUUM')
        outdb_conn.close()

run()
