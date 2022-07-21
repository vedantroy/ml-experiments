#! /usr/bin/env python3

from math import ceil
from pathlib import Path
import shutil
import argparse

from fastargs.decorators import param, get_current_config
from tqdm import tqdm
from composer.datasets.streaming import StreamingDatasetWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME

from dataset_writer_utils import CenterCropImageCaptionDataset, save_tensor, init_cli_args, init_dataset_args, init_dataset_dir

@param("files.imgs_dir")
@param("files.tags_db")
@param("files.dataset_dir")
def run(imgs_dir, tags_db, dataset_dir):
    init_dataset_dir()

    ds = CenterCropImageCaptionDataset(tags_db=tags_db, img_buckets_dir=imgs_dir)
    print(f"samples: {len(ds)}")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=12)


    FIELDS = ["img", "tags", "embeddings", "masks", "tokens"]
    with StreamingDatasetWriter(
        dataset_dir, FIELDS, shard_size_limit=1 << 24
    ) as writer:
        for batch in tqdm(dl):
            img, tags = batch["img"], batch["tags"]
            img = img[0]

            # IMPORTANT: By training on the raw tensors,
            # you must use the *exact* same Pytorch version
            # that you used to process the images
            # Anything else will cause data drift (I think ???)
            img_bytes = save_tensor(img)

            print(tags)

            # TODO: Might not need to return masks:
            embeds, masks = t5_encode_text(
                # function expects an array of strings
                tags,
                name=DEFAULT_T5_NAME,
                return_attn_mask=True,
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


init_dataset_args()
init_cli_args()
run()
