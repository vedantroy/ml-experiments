import torch.nn.functional as F

def convert_mask_to_ground_truth(masks, n_classes):
    return F.one_hot(masks, n_classes).permute(0, 3, 1, 2).float()