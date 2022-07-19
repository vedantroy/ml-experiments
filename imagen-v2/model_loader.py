from pathlib import Path

import torch
from imagen_pytorch import Imagen, BaseUnet64, ImagenTrainer

def create_trainer(params):
    unets = []
    image_sizes = []
    for unet in params["unets"]:
        typ, img_size = unet["type"] , unet["image_size"]
        image_sizes.append(img_size)
        if typ == "BaseUnet64":
            unets.append(BaseUnet64(**unet["params"]))
        else:
            raise Exception(f"Unknown unet: {type}")

    imagen = Imagen(
        unets=unets,
        image_sizes=image_sizes,
        **params["imagen"],
    )
    device = torch.device("cuda:0")
    cuda_imagen = imagen.to(device)
    trainer = ImagenTrainer(cuda_imagen, **params["trainer"])
    return trainer


def get_step(p):
    dir_name = str(p).split("/")[-1]
    return int(dir_name)

def get_checkpoint_paths(checkpoint_dir, run_id):
    if not run_id:
        return []

    print("Checkpoint dir: ", checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f"Checkpoint dir: {checkpoint_dir} does not exist"

    print(f"Loading state for run id: {run_id}")
    checkpoints = list((checkpoint_dir / run_id).glob("*"))

    checkpoints.sort(key=get_step)
    return checkpoints

def load_checkpoint_params_and_ctx(checkpoint_path):
    params = torch.load(checkpoint_path / "params.ckpt")
    ctx = torch.load(checkpoint_path / "context.ckpt")
    return params, ctx