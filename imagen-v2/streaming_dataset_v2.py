import torch
from composer.datasets.streaming import StreamingDataset
import io

def load_tensor(bytes):
    buf = io.BytesIO(bytes)
    return torch.load(buf)

class ImageWithCaptions(StreamingDataset):
    def __init__(self,
                 local: str,
                 shuffle: bool,
                 batch_size: int
                ) -> None:
        decoders = {
            'img': load_tensor,
            'tags': lambda data: data.decode('utf-8'),
            'embeddings': load_tensor,
            'masks': load_tensor,
            'tokens': lambda data: int(data),
        }
        super().__init__(local=local, remote=None, shuffle=shuffle, decoders=decoders, batch_size=batch_size)

    def __getitem__(self, i:int):
        obj = super().__getitem__(i)
        return obj