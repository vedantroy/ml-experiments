# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer example dataset."""

import itertools
import random

import numpy as np

from params import constants


def _infinite_shuffle(iterable, buffer_size):
    """Infinitely repeat and shuffle data from iterable."""
    # cycle('ABC') => A B C A B C A B C
    # https://docs.python.org/3/library/itertools.html#itertools.cycle
    ds = itertools.cycle(iterable)
    buf = [next(ds) for _ in range(buffer_size)]
    # QUESTION: Is this necessary??
    random.shuffle(buf)
    while True:
        item = next(ds)
        # `random.randint` is inclusive
        idx = random.randint(0, buffer_size - 1)
        result, buf[idx] = buf[idx], item
        yield result


class AsciiDataset:
    """In-memory dataset of a single-file ASCII dataset."""

    def __init__(self, path: str, batch_size: int, sequence_length: int):
        """Load a single-file ASCII dataset in memory."""
        # https://stackoverflow.com/questions/27679137/what-does-256-means-for-128-unique-characters-in-ascii-table
        # 128 ASCII characters that are actually used
        self.vocab_size = constants["vocab_size"]
        self._batch_size = batch_size

        crop_len = sequence_length + 1
        self.crop_len = crop_len

        with open(path, "r") as f:
            corpus = f.read()

        if not corpus.isascii():
            raise ValueError("Loaded corpus is not ASCII.")

        if "\0" in corpus:
            # Reserve 0 codepoint for pad token.
            raise ValueError("Corpus must not contain null byte.")

        # Tokenize by taking ASCII codepoints.
        corpus = np.array([ord(c) for c in corpus]).astype(np.int32)

        # Double-check ASCII codepoints.
        assert np.min(corpus) > 0
        assert np.max(corpus) < self.vocab_size

        num_batches, ragged = divmod(corpus.size, batch_size * crop_len)
        if ragged:
            corpus = corpus[:-ragged]
        assert corpus.shape == (num_batches * (batch_size * crop_len),)
        corpus = corpus.reshape([-1, crop_len])
        # (I think) at this point, our corpus is a bunch of sentences of length `crop_len`
        assert corpus.shape == (num_batches * batch_size, crop_len)

        if num_batches < 10:
            raise ValueError(
                f"Only {num_batches} batches; consider a shorter "
                "sequence or a smaller batch."
            )

        self._ds = _infinite_shuffle(corpus, batch_size * 10)
        # Note: Processing `batches_per_epoch` will not be a single pass
        # through the training data set since, roughly, each batch is randomly sampled
        # from the entire training dataset (may be overlap)
        self.batches_per_epoch = num_batches

    def __next__(self):
        """Yield next mini-batch."""
        batch = [next(self._ds) for _ in range(self._batch_size)]
        batch = np.stack(batch)
        assert batch.shape == (self._batch_size, self.crop_len)

        sentences_without_last_tok = batch[:, :-1]
        sentences_without_first_tok = batch[:, 1:]

        # Create the language modeling observation/target pairs.
        return dict(obs=sentences_without_last_tok, target=sentences_without_first_tok)

    def __iter__(self):
        return self
