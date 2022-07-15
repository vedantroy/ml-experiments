from pathlib import Path
import argparse

from fastargs import Section, Param, get_current_config
from fastargs.decorators import param
from imagen_pytorch import Imagen, BaseUnet64, ImagenTrainer
import torch
import torchvision.transforms.functional as T
import torchvision

Section("run", "run configuration").params(
    run_id=Param(
        str,
        "the run id",
    )
)

def generate_for_checkpoint(checkpoint):
    step = str(checkpoint).split("/")[-1]
    print(f"Generating for checkpoint: {checkpoint} (step = {step})")
    unet1 = BaseUnet64(
        # TODO: Fix this
        dim=256,
    )

    imagen = Imagen(
        unets=(unet1,),
        image_sizes=(64,),
        # If True, this just does 0 to 1 => -1 to 1
        auto_normalize_img=True,
    ).cuda()
    trainer = ImagenTrainer(imagen)

    print("Loading trainer...")
    trainer.load(checkpoint / "trainer.ckpt", strict=True)
    print("Trainer loaded")
    trainer.eval()

    sample = "1girl blush gift hair_ribbon looking_at_viewer object_hug pink_eyes pink_hair pleated_skirt ribbon school_uniform short_twintails skirt smile solo twintails"
    imgs = trainer.sample(texts=[sample], cond_scale=3.0, stop_at_unet_number=1)
    i = 0
    for img in imgs:
        pil = T.to_pil_image(img.cpu(), mode="RGB")
        pil.save(f"img_{step}_{i}.png", "PNG")
        i += 1

@param("run.run_id")
def run(run_id):
    print(f"Doing inference on run: {run_id}")

    checkpoints = list(Path(f"./runs/{run_id}").glob("*"))

    def get_step(p):
        dir_name = str(p).split("/")[-1]
        return int(dir_name)

    checkpoints.sort(key=get_step)

    for checkpoint in checkpoints:
        generate_for_checkpoint(checkpoint)

parser = argparse.ArgumentParser(description="Generate samples for a model")
config = get_current_config()
config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

run()
