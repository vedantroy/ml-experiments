from torch.utils.data import DataLoader
from streaming_dataset_v2 import ImageWithCaptions
from tqdm import tqdm

# API critique:
# - support truly parallel? when writing
# - tell me if local dir doesn't exist
# - Why do I need a batch_size/shuffle if this is a dataset ??
# (Why am I passing dataloader args to a dataset)
# I am a goon, plz make my life easier not harder :((
# e.g don't silently skip fields unless you set "SKIP_SILENT=True"

print("local arg")
dataset = ImageWithCaptions(local="./data/danbooru/artifacts/shards2", shuffle=False, batch_size=4)
dl = DataLoader(dataset, batch_size=4, num_workers=12)

for batch in tqdm(dl):
    pass
