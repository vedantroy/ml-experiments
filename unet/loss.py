import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, epsilon: float):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)
    # https://github.com/milesial/Pytorch-UNet/issues/319
    # If both predicted/target sets are empty, we return
    # a dice score of 1
    if sets_sum == 0:
        # We can't just do `return 1`
        # b/c we need to return a tensor w/ a `backward` function
        sets_sum = 2 * inter

    # not sure what the purpose of epsilon is 
    # I think it helps smooth the derivative &
    # is supposed to handle case where denominator is 0
    # altho we already return 1 in that case?
    # https://forums.fast.ai/t/understanding-the-dice-coefficient/5838
    return (2 * inter + epsilon) / (sets_sum + epsilon)

def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon: float):
    # Average of Dice coefficient for all classes
    assert input.shape == target.shape
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.shape == target.shape
    return 1 - multiclass_dice_coeff(input, target, 1e-6)