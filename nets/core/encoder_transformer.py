import torch.nn
from typing import Optional, Tuple

from nets.core.encoder_interface import EncoderInterface, \
    Conv2dSubsampling, PositionalEncoding, make_pad_mask


class Transformer(EncoderInterface):
    def __init__(self,
                 input_size: int = 80,
                 output_dim: int = 256,
                 attention_dim: int = 256,
                 attention_heads: int = 4,
                 feedforward_size: int = 2048,
                 num_layers: int = 12,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 global_cmvn: torch.nn.Module = None
                 ) -> None:

        super().__init__()

        self.input_size = input_size
        self.output_dim = output_dim
        self.Subsampling = Conv2dSubsampling(input_size, attention_dim)

        self.pos = PositionalEncoding(attention_dim, dropout_rate)

        self.global_cmvn = global_cmvn

        layer = TransformerLayer(
            attention_dim=attention_dim,
            nhead=attention_heads,
            feedforward_dim=feedforward_size,
            dropout=dropout_rate,
            normalize_before=normalize_before,
        )

        if normalize_before:
            norm = torch.nn.LayerNorm(attention_dim)
        else:
            norm = None

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=norm,
        )

        # self.output_layer = torch.nn.Sequential(
        #     torch.nn.Dropout(dropout_rate), torch.nn.Linear(attention_dim, output_dim)
        # )

    def forward(self,
                x: torch.Tensor,
                x_lens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.global_cmvn is not None:
            x = self.global_cmvn(x)

        x = self.Subsampling(x)
        x = self.pos(x)
        x = x.permute(1, 0, 2)  # (b, t, f) -> (t, b, f)

        #lengths = ((x_lens - 1) // 2 - 1) // 2
        lengths = (((x_lens - 1) >> 1) - 1) >> 1

        assert x.size(0) == lengths.max().item()

        mask = make_pad_mask(lengths)
        x = self.encoder(x, src_key_padding_mask=mask)

        #logits = self.output_layer(x)
        logits = logits.permute(1, 0, 2)  # (t, b, f) -> (b, t, f)

        return logits, lengths


class TransformerLayer(torch.nn.Module):

    def __init__(self,
                 attention_dim: int = 256,
                 nhead: int = 4,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 normalize_before: bool = True,
                 ) -> None:
        super(TransformerLayer, self).__init__()

        self.self_attention = torch.nn.MultiheadAttention(attention_dim, nhead, dropout=0.0)
        self.feedforward1 = torch.nn.Linear(attention_dim, feedforward_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.feedforward2 = torch.nn.Linear(feedforward_dim, attention_dim)

        self.norm1 = torch.nn.LayerNorm(attention_dim)
        self.norm2 = torch.nn.LayerNorm(attention_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu
        else:
            raise RuntimeError(
                "activation should be relu/gelu, not {}".format(activation)
            )

        self.normalize_before = normalize_before

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        residual = src

        if self.normalize_before:
            x = self.norm1(src)

        x2 = self.self_attention(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]

        x = residual + self.dropout1(x2)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x

        if self.normalize_before:
            x = self.norm2(x)

        x2 = self.feedforward2(self.dropout(self.activation(self.feedforward1(x))))
        x = residual + self.dropout2(x2)

        if not self.normalize_before:
            x = self.norm2(x)

        return x











