from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TVF

# ~ 8 minutes for 66848 images w/ 12 workers
dataset = ImageFolder(
    root="./data/danbooru/raw/valid_imgs",
    transform=lambda x: TVF.center_crop(TVF.pil_to_tensor(x), (256, 256)),
)
print("Dataset size:", len(dataset))
data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=12)

i = 0
for batch in tqdm(data_loader):
    i += 1
