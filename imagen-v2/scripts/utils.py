import io
import argparse

from fastargs.decorators import get_current_config
import torch

def save_tensor(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()

def init_cli_args(description):
    parser = argparse.ArgumentParser(description)
    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    config.validate(mode="stderr")
    config.summary()