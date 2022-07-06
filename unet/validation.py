import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from loss import dice_loss
from utils import convert_mask_to_ground_truth


def evaluate(model, dataloader, device):
    model.eval()
    num_val_batches = len(dataloader)
    loss_dice = 0
    loss_cross_entropy = 0

    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(
        dataloader,
        desc="Validation round",
        unit="batch",
        # remove the progress bar once it is finished
        leave=False,
    ):
        images, masks = batch["image"], batch["mask"]
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long)

        ground_truth = convert_mask_to_ground_truth(masks, model.n_classes)
        with torch.no_grad():
            masks_pred = model(images)
            probs = F.softmax(masks_pred, dim=1).float()

            loss_dice += dice_loss(probs, ground_truth)
            loss_cross_entropy += criterion(masks_pred, ground_truth)

    model.train()
    return (
        loss_dice / num_val_batches,
        loss_cross_entropy / num_val_batches,
        (loss_dice + loss_cross_entropy) / num_val_batches,
    )
