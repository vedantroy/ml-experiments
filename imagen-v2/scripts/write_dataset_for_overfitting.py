from pathlib import Path

from tqdm import tqdm
from fastargs import Param, Section
from fastargs.decorators import param
from composer.datasets.streaming import StreamingDatasetWriter

from dataset_writer_utils import center_crop_image, init_cli_args, init_dataset_args, init_dataset_dir, save_tensor

init_dataset_args()

Section("config", "configuration").params(
    repeat=Param(int, "the # of times to repeat the single image", default=5_000),
)

@param("files.dataset_dir")
@param("config.repeat")
def run(dataset_dir, repeat):
    init_dataset_dir()
    img_path = "./overfit.jpg"
    caption = "1girl blush gift hair_ribbon looking_at_viewer object_hug pink_eyes pink_hair pleated_skirt ribbon school_uniform short_twintails skirt smile solo twintails"

    caption = caption.encode("utf-8")
    cropped = center_crop_image(img_path, "sample_img", 0)
    assert cropped != None
    img_bytes = save_tensor(cropped)

    FIELDS = ["img", "tags"]
    with StreamingDatasetWriter(
        dataset_dir, FIELDS, shard_size_limit=1 << 24
    ) as writer:
        for _ in tqdm(range(repeat)):
            writer.write_sample(
                {
                    "img": img_bytes,
                    "tags": caption,
                },
            )

init_cli_args()
run()