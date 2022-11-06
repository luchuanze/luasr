
import math
import warnings
from typing import Optional, Tuple

import torch
from nets.core.encoder_interface import EncoderInterface
from nets.core.encoder_transformer import Transformer, make_pad_mask


class Conformer(Transformer):

    def __init__(self,
                 input_size: int = 80,
                 output_dim: int = 256,
                 attention_dim: int = 256,
                 attention_heads: int = 4,
                 feedforward_size: int = 2048,
                 num_layers: int = 12,
                 dropout_rate: float = 0.1,
                 cnn_module_kernel: int = 31,
                 normalize_before: bool = True,
                 global_cmvn: torch.nn.Module = None
                 ) -> None:
        super(Conformer, self).__init__(
            input_size=input_size,
            output_dim=output_dim,
            attention_dim=attention_dim,
            feedforward_size=feedforward_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            global_cmvn=global_cmvn
        )
        self.pos = RelPositionalEncoding(attention_dim, dropout_rate)

        layer = ConformerLayer(
            attention_dim,
            attention_heads,
            feedforward_size,
            dropout_rate,
            cnn_module_kernel,
            normalize_before,
        )

        self.encoder = ConformerEncoder(layer, num_layers)
        self.normalize_before = normalize_before
        if normalize_before:
            self.after_norm = torch.nn.LayerNorm(attention_dim)

    def forward(self,
                x: torch.Tensor,
                x_len: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.global_cmvn is not None:
            x = self.global_cmvn(x)

        x = self.Subsampling(x)

        x, pos_emb = self.pos(x)
        x = x.permute(1, 0, 2)  # (b, t, f) -> (t, b, f)

        lengths = (((x_len - 1) >> 1) - 1) >> 1
        assert x.size(0) == lengths.max().item()

        mask = make_pad_mask(lengths)

        x = self.encoder(x, pos_emb, x_key_padding_mask=mask)

        if self.normalize_before:
            x = self.after_norm(x)

        logits = self.output_layer(x)
        logits = logits.permute(1, 0, 2)  # (t, b, f) -> (b, t, f)

        return logits, lengths


class ConformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self,
                encoder_layer: torch.nn.Module,
                num_layers: int,
                norm: torch.nn.Module = None
                ) -> None:

        super(ConformerEncoder, self).__init__(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=norm
        )

    def forward(self,
                x: torch.Tensor,
                pos_emb: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                x_key_padding_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        output = x
        for layer in self.layers:
            output = layer(
                output,
                pos_emb,
                x_mask=mask,
                x_key_padding_mask=x_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class ConformerLayer(torch.nn.Module):
    def __init__(self,
                attention_dim: int,
                nhead: int,
                feedforward_dim: int = 2048,
                dropout: float = 0.1,
                cnn_module_kernel: int = 31,
                normalize_before: bool = True,
                ) -> None:
        super(ConformerLayer, self).__init__()
        self.self_attention = RelPositionMultiHeadAttention(
            attention_dim, nhead, dropout_rate=0.0
        )

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(attention_dim, feedforward_dim),
            Swish(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward_dim, attention_dim),
        )

        self.feed_forward_macaron = torch.nn.Sequential(
            torch.nn.Linear(attention_dim, feedforward_dim),
            Swish(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward_dim, attention_dim)
        )

        self.conv_module = ConvolutionModule(attention_dim, cnn_module_kernel)

        self.norm_ff_macaron = torch.nn.LayerNorm(attention_dim)
        self.norm_ff = torch.nn.LayerNorm(attention_dim)
        self.norm_mha = torch.nn.LayerNorm(attention_dim)

        self.ff_scale = 0.5
        self.norm_conv = torch.nn.LayerNorm(attention_dim)
        self.norm_final = torch.nn.LayerNorm(attention_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(self,
                x: torch.Tensor,
                pos_emb: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None,
                x_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        residual = x
        if self.normalize_before:
            x = self.norm_ff_macaron(x)

        x = residual + self.ff_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        if not self.normalize_before:
            x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att = self.self_attention(
            x,
            x,
            x,
            pos_emb=pos_emb,
            key_padding_mask=x_key_padding_mask,
            attention_mask=x_mask,
        )[0]

        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        residual = x
        if self.normalize_before:
            x = self.norm_conv(x)
        x = self.conv_module(x)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = self.feed_forward(x)
        x = residual + self.ff_scale * self.dropout(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.normalize_before:
            x = self.norm_final(x)

        return x


class ConvolutionModule(torch.nn.Module):
    """
    ConvolutionModule in Conformer model.
    """

    def __init__(self,
                channels: int,
                kernel_size: int,
                bias: bool = True
                ) -> None:

        super(ConvolutionModule, self).__init__()

        # Kernerl size should be a odd number for 'SAME' padding
        assert (kernel_size -1) % 2 == 0

        self.pointwise_conv1 = torch.nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias
        )

        self.norm = torch.nn.LayerNorm(channels)

        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.activation = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: input tensor of shape(t, b, f)
        :return:
        """
        x = x.permute(1, 2, 0)  # (b, f, t)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (b, 2*f, t)
        x = torch.nn.functional.glu(x, dim=1)  # (b, f, t)

        x = self.depthwise_conv(x)

        x = x.permute(0, 2, 1)
        self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)

        return x.permute(2, 0, 1)


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self,
                attention_dim: int,
                dropout_rate: float,
                max_len: int = 5000
                ) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.attention_dim = attention_dim
        self.scale = math.sqrt(self.attention_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None

    def extend_pe(self, x: torch.Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                    return

        pe_positive = torch.zeros(x.size(1), self.attention_dim)
        pe_negative = torch.zeros(x.size(1), self.attention_dim)

        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.attention_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.attention_dim)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.sin(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self,
                x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positional encoding
        :param x:
        A input tensor of shape(b, t, f).
        :return:
        input tensor with scale.
        encoded tensor (b, 2*t-1, f).
        """

        self.extend_pe(x)
        x = x * self.scale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1: self.pe.size(1) // 2
            + x.size(1),
            :
        ]

        return self.dropout(x), self.dropout(pos_emb)

# class RelPositionalEncoding(torch.nn.Module):
#     """
#     PE(pos, 2i) = sin(pos/ 10000^(2i/dim))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
#
#     1 / (10000^(2i/d_model)) = exp(-log(10000^(2i/d_model)))
#                                = exp(-1* 2i / d_model * log(100000))
#                                = exp(2i * -(log(10000) / d_model))
#     """
#
#     def __init__(self,
#                  dim: int,
#                  dropout: float = 0.1,
#                  ) -> None:
#
#         super().__init__()
#         self.dim = dim
#         self.scale = math.sqrt(self.dim)
#         self.dropout = torch.nn.Dropout(p=dropout)
#
#         self.pe = torch.zeros(1, 0, self.dim, dtype=torch.float32)
#
#     def extend_pe(self,
#                   x: torch.Tensor
#                   ) -> None:
#
#         if self.pe is not None:
#             if self.pe.size(1) >= x.size(1):
#                 self.pe = self.pe.to(dtype=x.dtype, device=x.device)
#                 return
#         pe = torch.zeros(x.size(1), self.dim, dtype=torch.float32)
#         position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, self.dim, 2, dtype=torch.float32)
#             * -(math.log(10000.0) / self.dim)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         # pe is of shape(1, t, dim) where t is x.size(1）
#         self.pe = pe.to(device=x.device, dtype=x.dtype)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Add positional encoding.
#         :param x:
#         Its shape is (b, t, c)
#         :return:
#         tensor of shape(b, t, c)
#         """
#         self.extend_pe(x)
#         x = x * self.scale
#         pos_emb = self.pe[:, : x.size(1), :]
#         return self.dropout(x), self.dropout(pos_emb)


class RelPositionMultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention layer with relative position encoding
    """

    def __init__(self,
                embed_dim: int,
                nhead: int,
                dropout_rate: float = 0.0
                ) -> None:

        super(RelPositionMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.dropout_rate = dropout_rate
        self.head_dim = embed_dim // nhead
        assert (
            self.head_dim * nhead == self.embed_dim
        ), "embed_dim must be divisible by nhead"

        self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

        self.linear_pos = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = torch.nn.Parameter(torch.Tensor(nhead, self.head_dim))
        self.pos_bias_v = torch.nn.Parameter(torch.Tensor(nhead, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.in_proj.weight)
        torch.nn.init.constant_(self.in_proj.bias, 0.0)
        torch.nn.init.constant_(self.out_proj.bias, 0.0)

        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                pos_emb: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                needs_weights: bool = True,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """

        :param query: (l, b, e)
        :param key:  (s, b, e)
        :param value: (s, b, e)
        :param pos_emb: (b, 2*l-1, e)
        :param key_padding_mask: (b, s)
        :param needs_weights:
        :param attention_mask: (l, s)
        :return:
        attention_output (l, b, e)
        attention_output_weights (b, l, s)
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
            self.embed_dim,
            self.nhead,
            self.in_proj.weight,
            self.in_proj.bias,
            self.dropout_rate,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=needs_weights,
            attention_mask=attention_mask
        )

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape
        assert n == 2 * time1 - 1
        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, time1, time1),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def multi_head_attention_forward(self,
                                     query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     pos_emb: torch.Tensor,
                                     embed_dim_to_check: int,
                                     num_heads: int,
                                     in_proj_weight: torch.Tensor,
                                     in_proj_bias: torch.Tensor,
                                     dropout_rate: float,
                                     out_proj_weight: torch.Tensor,
                                     out_proj_bias: torch.Tensor,
                                     training: bool = True,
                                     key_padding_mask: Optional[torch.Tensor] = None,
                                     need_weights: bool = True,
                                     attention_mask: Optional[torch.Tensor] = None,
                                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        q_len, batch, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = torch.nn.functional.linear(
                query, in_proj_weight, in_proj_bias
            ).chunk(3, dim=-1)
            # self-attention
        else:
            raise RuntimeError(
                "This is not self attention"
            )

        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(q_len, batch, num_heads, head_dim)
        k = k.contiguous().view(-1, batch, num_heads, head_dim)
        v = v.contiguous().view(-1, batch * num_heads, head_dim).transpose(0, 1)

        k_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == batch
            assert key_padding_mask.size(1) == k_len

        q = q.transpose(0, 1)  # (b, t1, h, d_k)

        pos_emb_batch = pos_emb.size(0)  # actually it is 1

        p = self.linear_pos(pos_emb).view(pos_emb_batch, -1, num_heads, head_dim)
        p = p.transpose(1, 2)  # (b, h, 2*t1 - 1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)  # (b, h, t1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        k = k.permute(1, 2, 3, 0)  # （b, h, d_k, t2）
        mat_ac = torch.matmul(q_with_bias_u, k)  # (b, h, t1, t2)
        mat_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (b, h, t1, 2*t1 -1)
        mat_bd = self.rel_shift(mat_bd)

        attention_score = (mat_ac + mat_bd) * scaling  # (b, h, t1, t2)
        attention_score = attention_score.view(batch * num_heads, q_len, -1)
        assert list(attention_score.size()) == [batch * num_heads, q_len, k_len]

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_score.masked_fill_(attention_mask, float("-inf"))
            else:
                attention_score += attention_mask

        if key_padding_mask is not None:
            attention_score = attention_score.view(batch, num_heads, q_len, k_len)
            attention_score = attention_score.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )
            attention_score = attention_score.view(batch * num_heads, q_len, k_len)

        attention_score = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_score = torch.nn.functional.dropout(
            attention_score, p=dropout_rate, training=training
        )

        attention_output = torch.bmm(attention_score, v)
        assert list(attention_output.size()) == [batch * num_heads, q_len, head_dim]
        attention_output = (
            attention_output.transpose(0, 1)
            .contiguous()
            .view(q_len, batch, embed_dim)
        )

        attention_output = torch.nn.functional.linear(
            attention_output, out_proj_weight, out_proj_bias
        )

        return attention_output


class Swish(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)




        








