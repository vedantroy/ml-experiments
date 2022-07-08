import torch
import torch.nn as nn
from helpers import assert_shape

# NOTATION:
# W_k = W (k as subscript)
# Wk = W (k as superscript)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask, params):
        super().__init__()

        self.params = params
        self.d_model = d_model
        # This is a buffer we need, but we don't want it to be optimized
        # by the optimizer
        self.register_buffer("mask", mask)
        d_k = d_model / num_heads
        assert d_k.is_integer()
        self.d_k = int(d_k)
        self.num_heads = num_heads

        # TODO: Why no biases?
        # Notice, there are no biases:
        # MultiHead(Q, K, V) = Concat(head_1, ..., head_h)(Wo)
        #  where head_i = Attention(QWq, KWk, VWv)

        # A confusing part: there should be multiple attention heads
        # each with its own copy of Wq, Wk, Wv -- but to represent that
        # we'll just use a single giant matrix + Pytorch trickery
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        orig_shape = x.shape
        batch_size, sequence_len, d_model = x.shape

        assert self.params["batch_size"] == batch_size
        assert self.params["sequence_len"] == sequence_len
        assert self.d_model == d_model

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        expected_shape = (batch_size, sequence_len, d_model)
        assert_shape((Q, K, V), expected_shape)

        Q, K, V = (
            self.split_into_heads(Q),
            self.split_into_heads(K),
            self.split_into_heads(V),
        )

        expected_shape = (batch_size, self.num_heads, sequence_len, self.d_k)
        assert_shape((Q, K, V), expected_shape)

        K_T = K.transpose(2, 3)
        assert K_T.shape == (batch_size, self.num_heads, self.d_k, sequence_len)
        # For high-dimensional tensors, the matrix multiplication can only be
        # operated on the last two dimensions, which requires the previous dimensions to be equal.
        query_attention_to_keys = Q @ K_T
        query_attention_to_keys *= 1 / (self.d_k**0.5)

        assert query_attention_to_keys.shape == (
            batch_size,
            self.num_heads,
            sequence_len,
            sequence_len,
        )
        assert self.mask.shape == (sequence_len, sequence_len)

        # From paper:
        # > We need to prevent leftward
        # > information flow in the decoder to preserve the auto-regressive property. We implement this
        # > inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
        query_attention_to_keys.masked_fill_(self.mask == 0, -1e9)
        query_attention_to_keys_normalized = torch.softmax(
            query_attention_to_keys, dim=3
        )

        combined_value_vectors = query_attention_to_keys_normalized @ V
        assert_shape(combined_value_vectors, expected_shape)
        transposed = combined_value_vectors.transpose(1, 2)
        assert_shape(transposed, (batch_size, sequence_len, self.num_heads, self.d_k))
        concatted = transposed.reshape(batch_size, sequence_len, d_model)

        out = self.linear(concatted)
        assert_shape(out, orig_shape)
        return out

    def split_into_heads(self, tensor):
        batch_size, sequence_len, d_model = tensor.shape
        assert self.d_k * self.num_heads == d_model
        return tensor.view(
            batch_size, sequence_len, self.num_heads, self.d_k
        ).transpose(1, 2)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        widening_factor: int,
        mask: torch.Tensor,
        params,
    ):
        super().__init__()
        self.params = params
        self.d_ff = widening_factor * d_model

        self.attention = MultiHeadAttention(d_model, num_heads, mask, params)

        # From paper:
        # > We apply dropout [33] to the output of each sub-layer, before it is added to the
        # > sub-layer input and normalized.
        self.dropout1 = nn.Dropout(p=params["dropout"])
        # TODO: How was this epsilon chosen?
        # TODO: Why would we do layer norm without learnable parameters? (`elementwise_affine=False`)
        # TODO: We need layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)

        # https://stats.stackexchange.com/questions/485910/what-is-the-role-of-feed-forward-layer-in-transformer-neural-network-architectur
        self.lin1 = nn.Linear(d_model, self.d_ff)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.d_ff, d_model)

        self.dropout2 = nn.Dropout(p=params["dropout"])
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        batch_size, sequence_len, d_model = (
            self.params["batch_size"],
            self.params["sequence_len"],
            self.params["d_model"],
        )

        original_x = x
        x = self.attention(x)
        assert x.shape == (batch_size, sequence_len, d_model)

        # add & normalize
        x = self.norm1(self.dropout1(original_x + x))
        assert x.shape == (batch_size, sequence_len, d_model)

        original_x = x
        x = self.lin1(x)
        assert x.shape == (batch_size, sequence_len, self.d_ff)
        x = self.relu(x)
        x = self.lin2(x)
        assert x.shape == (batch_size, sequence_len, d_model)

        # add & normalize again
        x = self.norm2(self.dropout2(original_x + x))
        return x


# class PositionalEncoding(nn.Module):
#
#    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#        super().__init__()
#        self.dropout = nn.Dropout(p=dropout)
#
#        position = torch.arange(max_len).unsqueeze(1)
#        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#        pe = torch.zeros(max_len, 1, d_model)
#        pe[:, 0, 0::2] = torch.sin(position * div_term)
#        pe[:, 0, 1::2] = torch.cos(position * div_term)
#        self.register_buffer('pe', pe)
#
#    def forward(self, x: Tensor) -> Tensor:
#        """
#        Args:
#            x: Tensor, shape [seq_len, batch_size, embedding_dim]
#        """
#        x = x + self.pe[:x.size(0)]
#        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        d_model: int,
        widening_factor: int,
        sequence_len: int,
        layers: int,
        mask: torch.Tensor,
        params,
    ):
        super().__init__()

        self.params = params
        self.vocab_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        # self.positional_embedding = torch.zeros((sequence_len, d_model))
        # self.register_buffer('positional_embedding', self.positional_embedding)

        self.positional_embedding = nn.parameter.Parameter(
            torch.zeros((sequence_len, d_model)), requires_grad=True
        )

        # Our `initialize_weights` function doesn't cover this for some reason
        nn.init.kaiming_uniform_(self.positional_embedding)

        self.embedding_dropout = nn.Dropout(p=params["dropout"])

        assert self.vocab_embedding.weight.shape == (vocab_size, d_model)
        assert self.positional_embedding.shape == (sequence_len, d_model)

        self.decoder_layers = nn.ModuleList(
            DecoderLayer(num_heads, d_model, widening_factor, mask, params)
            for _ in range(layers)
        )

        # Maps the output embeddings back to tokens
        # You could also do:
        #     torch.matmul(decoder_output, self.vocab_embedding.weight.tranpose(0, 1))
        # inside of `forward` if you wanted
        # If you wanted to re-use the input embedding matrix
        # TODO: How does gradient flow work if we re-use the embedding matrix?
        # TODO: Why don't we need to subtract positional encodings if using tied?
        # https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/model/transformer.py#L292

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, sequence_len = x.shape

        assert self.params["batch_size"] == batch_size
        assert self.params["sequence_len"] == sequence_len

        embeddings = self.vocab_embedding(x)
        assert embeddings.shape == (batch_size, sequence_len, self.params["d_model"])
        embeddings_with_positions = embeddings + self.positional_embedding
        embeddings_with_positions = self.embedding_dropout(embeddings_with_positions)

        # From paper
        # > we apply dropout to the sums of the embeddings and the positional encodings

        decoder_output = embeddings_with_positions
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output)

        return self.linear(decoder_output)
