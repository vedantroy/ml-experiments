from PIL import Image
import torchvision.transforms.functional as T

from files import files

img_files = files
for i, file in enumerate(img_files):
    print(i)
    img = None
    try:
        img = Image.open(file)
    except Exception:
        continue

    try:
        img = T.pil_to_tensor(img)
    except Exception:
        continue