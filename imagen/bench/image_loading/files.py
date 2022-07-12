from pathlib import Path

imgs_path = "/home/vedant/Desktop/ml-experiments/imagen/data/danbooru/data/dataset/imgs/0000/"
imgs_path = Path(imgs_path)
files = list(imgs_path.glob("*"))
N = 500
files = files[:N]