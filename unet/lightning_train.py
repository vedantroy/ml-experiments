from importlib.resources import is_resource
from pathlib import Path
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from model import UNet
import wandb
from unet.data import BasicDataset


from unet.loss import dice_loss
from unet.utils import convert_mask_to_ground_truth


def get_losses(batch, model, n_classes):
    imgs, masks = batch["image"], batch["mask"]

    # imgs = imgs.to(device=self.device, dtype=torch.float32)
    # masks = masks.to(device=self.device, dtype=torch.long)

    mask_preds = model(imgs)
    probs = F.softmax(mask_preds, dim=1).float()
    one_hots = convert_mask_to_ground_truth(masks, n_classes)

    loss_cross_entropy = F.cross_entropy(mask_preds, masks)
    loss_dice = dice_loss(probs, one_hots)
    combined_loss = loss_cross_entropy + loss_dice
    return (loss_cross_entropy, loss_dice, combined_loss, (imgs, masks, mask_preds))


class LightningModel(pl.LightningModule):
    def __init__(self, scale: float, percent_of_data_used_for_validation: int, batch_size: int, wandb_logger, run_id: str):
        super().__init__()
        self.model = UNet(n_in_channels=3, n_classes=2)
        dataset = BasicDataset(
            images_dir="./data/imgs", masks_dir="./data/masks", scale=scale
        )
        n_val = int(len(dataset) * (percent_of_data_used_for_validation / 100))
        n_train = len(dataset) - int(n_val)
        assert n_train + n_val == len(dataset)

        train_set, val_set = random_split(
                dataset, [n_train, n_val], 
                # if we use seed_everything, do we still need this?
                #generator=torch.Generator().manual_seed(0)
        )
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.loader_args = dict(
            batch_size=batch_size,
            num_workers=4,
            # Only useful if you're training on GPU
            pin_memory=self.device.type == "cuda",
        )
        self.sample_arg = None
        self.wandb_logger = wandb_logger
        self.run_id = run_id
        self.wandb_logger = wandb_logger

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        # no LR schedulers
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        cross_entropy , dice, combined, _ = get_losses(batch, self, self.model.n_classes)
        return combined

    def validation_step(self, batch, batch_idx): 
        cross_entropy , dice, combined, (imgs, masks, mask_preds) = get_losses(batch, self, self.model.n_classes)
        if not self.sample_arg:
            self.sample_arg = imgs[0]

        self.log("val_cross_entropy_loss", cross_entropy)
        self.log("val_dice_loss", dice)
        self.log("val_combined_loss", combined)
        wandb.log({
            "image": wandb.Image(imgs[0].cpu()),
            "masks": {
                "true": wandb.Image(masks[0].float().cpu()),
                "pred": wandb.Image(mask_preds.argmax(dim=1)[0].float().cpu())
            },
            "step": trainer.global_step
        })

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, **self.loader_args)

    def val_dataloader(self):
        loader =  DataLoader(self.val_set, shuffle=False, drop_last=True, **self.loader_args)
        assert len(loader) > 0, "validation loader must have enough data for at least 1 full batch"
        return loader

    def on_train_end(self):
        path = Path("./runs/{self.run_id}")
        path.mkdir(parents=True, exist_ok=True)

        model_path =  path / "final_model.pt"
        model_onnx_path = path / "final_model.onnx"
        torch.save(self.state_dict(), model_path)
        torch.onnx.export(self.model, self.sample_arg, model_onnx_path)
        wandb.save(model_path)
        wandb.save(model_onnx_path)

def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--id",
        "-R",
        type=str,
        default=False,
        help="Load model using an existing run id",
    )
    return parser.parse_args()

if __name__ == "__main__":
    model = run_id = config = None
    args = get_args()
    is_resuming = False
    if args.id:
        is_resuming = True
        run_id = args.id
        lighting_checkpoint = Path(f"./runs/{run_id}/lightning.checkpoint")
        wandb_checkpoint = Path(f"./runs/{run_id}/wandb.checkpoint")
        if not lighting_checkpoint.is_file():
            raise FileNotFoundError(f"Could not find checkpoint at {lighting_checkpoint}")
        model = LightningModel.load_from_checkpoint(lighting_checkpoint)
        config = torch.load(wandb_checkpoint)
    else:
        run_id = wandb.util.generate_id()
        print(f"Starting new run with id: {run_id}")
        config = dict(
            scale=0.5,
            percent_of_data_used_for_validation=10,
            batch_size=1,
            learning_rate=3e-4,
            amp=False,
            epochs=5,
        )

    p = Path(f"./runs/{run_id}")
    p.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(id=run_id)
    trainer = pl.Trainer(max_epochs=config['epochs'], logger=logger)
    model = LightningModel(**config, run_id=run_id)

    with wandb.init(project="UNet", id=run_id, resume="must" if is_resuming else "never", config=config):
        wandb.watch(model.model, criterion=None, log="all", log_freq=100)

        try:
            trainer.fit(model)
        except KeyboardInterrupt as e:
            throw = e

    trainer.save_checkpoint(f"./runs/{run_id}/lightning.checkpoint")
    torch.save(wandb.config, f"./runs/{run_id}/wandb.checkpoint")

    if throw:
        raise throw