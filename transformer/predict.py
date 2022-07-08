from pathlib import Path
import argparse

import torch
import torch.nn.functional as F

from model import Transformer

def get_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "id",
        type=str,
        default=False,
        help="Load model for predictions using an existing run id",
    )
    parser.add_argument(
        "sentence",
        type=str,
        default=False,
        help="The sentence to complete",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args("Auto-complete a sentence")
    run_id = args.id

    model_folder = Path(f"./runs/{run_id}")
    assert model_folder.is_dir(), f"{model_folder} must be a directory"

    config = torch.load(model_folder / "wandb.checkpoint")
    seq_len = config["sequence_len"]

    sentence = args.sentence
    assert len(sentence) < seq_len, f"Max sequence length is: {seq_len}"

    # Get the most recent model
    model_files = model_folder.glob("final_model-*.pt")
    model_files = [file.name for file in model_files if file.suffix == ".pt"]
    model_files.sort(key=lambda x: int(str(x)[len("final_model-"):-len(".pt")]))
    model_file = model_files[-1]
    model_state_dict = torch.load(model_folder / model_file)

    model = Transformer(
        vocab_size=config["vocab_size"],
        num_heads=config["heads"],
        d_model=config["d_model"],
        widening_factor=config["widening_factor"],
        sequence_len=seq_len,
        layers=config["layers"],
        mask=torch.tril(torch.ones((seq_len, seq_len))),
        params={
            "batch_size": 1,
            "sequence_len": seq_len,
            "d_model": config["d_model"],
            "dropout": config["dropout"],
        }
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    print(f"Auto-completing sentence: {sentence}")
    toks = torch.IntTensor([ord(c) for c in sentence])
    # right pad with zeroes
    chars_provided = len(toks)
    assert chars_provided == len(sentence)

    chars_to_complete = seq_len - chars_provided
    toks = F.pad(toks, (0, chars_to_complete))
    assert len(toks) == seq_len

    # Auto-regressive transformer
    # Do I even need this if the model is in eval mode?
    with torch.no_grad():
        for pos in range(chars_to_complete):
            # preds[0] = prediction for token after 1st token (which is the token @ the 0th idx)
            # preds[1] = prediction for token after 2nd token (which is the token @ the 1st idx)
            idx = chars_provided + pos - 1
            preds = model(toks.unsqueeze(0))
            assert preds.shape == (1, seq_len, config["vocab_size"])
            next_tok = preds[0, idx].argmax()
            toks[idx + 1] = next_tok

    toks = toks.tolist()
    toks = [chr(tok) for tok in toks]
    print("".join(toks))