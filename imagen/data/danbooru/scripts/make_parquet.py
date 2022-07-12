from pathlib import Path

# faster sqlite3 client
from supersqlite import sqlite3
from PIL import Image
import torchvision.transforms.functional as T
import pandas

conn = sqlite3.connect("../artifacts/tags.sqlite")

# Each shard file is <= 5 GB
SHARD_FILE_BYTES = 1e+9 * 5
IMG_BUCKETS_DIR = "../raw/imgs"

buckets_dir = Path(IMG_BUCKETS_DIR)
assert buckets_dir.is_dir()
img_files = list(buckets_dir.glob("*/*"))

for path in img_files:
    path = str(path)
    if (
        (not path.endswith("jpg"))
        and (not path.endswith("jpeg"))
        and (not path.endswith("png"))
    ):
        continue
    img = None
    try:
        # We can't use torchvision's read_image
        # b/c that silently crashes Python
        img = Image.open(path)
    except Exception:
        continue

    try:
        # PIL will load "truncated" images
        # that will crash here
        img = T.pil_to_tensor(img)
    except Exception:
        continue

    # Image is not RGB
    if img.shape[0] != 3:
        continue

    assert len(img.shape) == 3
    print("legit image")


rows = conn.execute("select * from tags limit 2").fetchall()
rows = conn.execute("select * from tags limit 2")
print(rows)
rows = conn.execute(f'select tag from tags where tags.id = "4160"').fetchall()
print(rows)