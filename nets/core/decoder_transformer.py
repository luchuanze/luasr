import torch.nn
from nets.core.encoder_interface import PositionalEncoding, make_pad_mask
from typing import Optional


class Transformer(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int = 4,
                 feedforward_size: int = 2048,
                 num_layers: int = 6,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True
                 ):

        super(Transformer, self).__init__()

        attention_dim = encoder_output_size
        self.attention_heads = attention_heads
        self.embed = torch.nn.Embedding(vocab_size, attention_dim)
        self.pos = PositionalEncoding(attention_dim, dropout_rate)

        self.normalize_before = normalize_before
        self.extra_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)

        if self.normalize_before:
            norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        else:
            norm = None

        layer = TransformerLayer(
            attention_dim=attention_dim,
            nhead=attention_heads,
            feedforward_dim=feedforward_size,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before
        )

        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers,
            norm=norm
        )

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_out_mask: torch.Tensor,
                y_pad: torch.Tensor,
                y_lens: torch.Tensor):

        tgt = y_pad
        tgt_mask = ~make_pad_mask(y_lens).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)

        #
        arrange = torch.arange(tgt_mask.size(-1), device=tgt.device)
        m = arrange.expand(tgt_mask.size(-1), tgt_mask.size(-1))
        arrange = arrange.unsqueeze(-1)
        m = m <= arrange
        m = m.unsqueeze(0)
        tgt_mask = tgt_mask & m

        tgt_nhead_mask = torch.repeat_interleave(~tgt_mask, self.attention_heads, dim=0)

        x = self.embed(tgt)
        x = self.pos(x)
        x = x.permute(1, 0, 2)  # (b, t, f) -> (t, b, f)
        encoder_out = encoder_out.permute(1, 0, 2)
        x = self.decoder(x, encoder_out,
                         tgt_mask=tgt_nhead_mask,
                         memory_key_padding_mask=encoder_out_mask)

        x = self.output_layer(x)

        x = x.permute(1, 0, 2)

        return x

    def forward_one_step(self,
                         encoder_out: torch.Tensor,
                         encoder_out_mask: torch.Tensor,
                         y: torch.Tensor,
                         y_mask: torch.Tensor):
        x = self.embed(y)
        x = self.pos(x)
        x = x.permute(1, 0, 2)
        encoder_out = encoder_out.permute(1, 0, 2)

        tgt_nhead_mask = torch.repeat_interleave(~y_mask, self.attention_heads, dim=0)
        x = self.decoder(x, encoder_out,
                         tgt_mask=tgt_nhead_mask,
                         memory_key_padding_mask=~encoder_out_mask)

        x = self.output_layer(x)
        x = x.permute(1, 0, 2)
        x = torch.log_softmax(x, dim=-1)
        x = x.view(y.size(0), -1)

        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 attention_dim: int = 256,
                 nhead: int = 4,
                 dropout_rate: float = 0.0,
                 feedforward_dim: int = 2048,
                 normalize_before: bool = True):
        super(TransformerLayer, self).__init__()

        self.self_attention = torch.nn.MultiheadAttention(
            attention_dim,
            num_heads=nhead,
            dropout=0.0
        )
        self.src_attention = torch.nn.MultiheadAttention(
            attention_dim,
            num_heads=nhead,
            dropout=0.0
        )

        self.feedforward = PositionwiseFeedForward(
            attention_dim,
            feedforward_dim,
            dropout_rate=dropout_rate
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.norm1 = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.norm3 = torch.nn.LayerNorm(attention_dim, eps=1e-5)

        self.normalize_before = normalize_before

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        tgt_q = tgt
        tgt_q_mask = tgt_mask

        x2 = self.self_attention(tgt_q, tgt, tgt, attn_mask=tgt_q_mask)[0]

        x = residual + self.dropout(x2)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x3 = self.src_attention(x, memory, memory, key_padding_mask=memory_key_padding_mask)[0]

        x = residual + self.dropout(x3)

        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feedforward(x))

        if not self.normalize_before:
            x = self.norm3(x)

        return x


class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self,
                 ndim: int,
                 inner_dim: int,
                 dropout_rate: float
                 ):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(ndim, inner_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear2 = torch.nn.Linear(inner_dim, ndim)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:

        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(self.dropout(x))