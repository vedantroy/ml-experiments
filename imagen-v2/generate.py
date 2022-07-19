from pathlib import Path
import argparse

from fastargs import Section, Param, get_current_config
from fastargs.decorators import param
import torchvision.transforms.functional as T

from model_loader import get_checkpoint_paths, load_checkpoint_params_and_ctx, create_trainer, get_step

Section("run", "run configuration").params(
    run_id=Param(
        str,
        "the run id",
    ),
    latest=Param(
        bool,
        "whether to only generate for the latest run",
        default=False
    )
)

def generate_for_checkpoint(checkpoint_path):
    params, _ = load_checkpoint_params_and_ctx(checkpoint_path)
    trainer = create_trainer(params)
    trainer.load(checkpoint_path / "trainer.ckpt")

    step = get_step(checkpoint_path)

    # sample = "sample_caption"
    sample = "1girl black_legwear brown_hair controller green_eyes hat joystick long_hair solo teruterubouzu thighhigh"
    imgs = trainer.sample(texts=[sample], cond_scale=10., stop_at_unet_number=1)
    i = 0
    for img in imgs:
        pil = T.to_pil_image(img.cpu(), mode="RGB")
        pil.save(f"./samples/img_{step}_{i}.png", "PNG")
        i += 1

@param("run.run_id")
@param("run.latest")
def run(run_id, latest):
    print(f"Doing inference on run: {run_id}")

    checkpoint_paths = get_checkpoint_paths("./runs", run_id)
    if latest:
        generate_for_checkpoint(checkpoint_paths[-1])
    else:
        for checkpoint in checkpoint_paths:
            generate_for_checkpoint(checkpoint)

parser = argparse.ArgumentParser(description="Generate samples for a model")
config = get_current_config()
config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

run()
