
import math
import warnings
from typing import Optional, Tuple

import torch
from nets.core.encoder_interface import EncoderInterface, RelPositionalEncoding
from nets.core.encoder_transformer import Transformer, make_pad_mask
from nets.core.attention import MultiHeadAttention, RelPositionMultiHeadAttention


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
        #x = x.permute(1, 0, 2)  # (b, t, f) -> (t, b, f)

        #lengths = (((x_len - 1) >> 1) - 1) >> 1
        #assert x.size(1) == lengths.max().item()

        mask_pad = ~make_pad_mask(x_len).unsqueeze(1)
        mask_pad = mask_pad[:,:,:-2:2][:,:,:-2:2]
        lengths = mask_pad.squeeze(1).sum(1).to(torch.int32)
        assert x.size(1) == lengths.max().item()

        x = self.encoder(x, pos_emb, x_key_padding_mask=mask_pad)

        if self.normalize_before:
            x = self.after_norm(x)

        #logits = self.output_layer(x)
        logits = x
        #logits = logits.permute(1, 0, 2)  # (t, b, f) -> (b, t, f)


        return logits, lengths, mask_pad


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


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


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

        # self.feed_forward = torch.nn.Sequential(
        #     torch.nn.Linear(attention_dim, feedforward_dim),
        #     torch.nn.SiLU(),
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(feedforward_dim, attention_dim),
        # )
        #
        # self.feed_forward_macaron = torch.nn.Sequential(
        #     torch.nn.Linear(attention_dim, feedforward_dim),
        #     torch.nn.SiLU(),
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(feedforward_dim, attention_dim)
        # )
        self.feed_forward = PositionwiseFeedForward(attention_dim, feedforward_dim, dropout, torch.nn.SiLU())
        self.feed_forward_macaron = PositionwiseFeedForward(attention_dim, feedforward_dim, dropout, torch.nn.SiLU())

        self.conv_module = ConvolutionModule(attention_dim, cnn_module_kernel)

        self.norm_ff_macaron = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.norm_ff = torch.nn.LayerNorm(attention_dim)
        self.norm_mha = torch.nn.LayerNorm(attention_dim)

        self.ff_scale = 0.5
        self.norm_conv = torch.nn.LayerNorm(attention_dim)
        self.norm_final = torch.nn.LayerNorm(attention_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before
        self.concat_linear = torch.nn.Linear(attention_dim + attention_dim, attention_dim)

    def forward(self,
                x: torch.Tensor,
                pos_emb: torch.Tensor,
                x_mask: Optional[torch.Tensor] = None,
                x_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        residual = x
        if self.normalize_before:
            x = self.norm_ff_macaron(x)

        x = self.feed_forward_macaron(x)

        x = residual + self.ff_scale * self.dropout(
            x
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
        )

        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        residual = x
        if self.normalize_before:
            x = self.norm_conv(x)
        x = self.conv_module(x, x_key_padding_mask)
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

        self.norm = torch.nn.BatchNorm1d(channels)

        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor, mask_pad: Optional[torch.Tensor] = None,) -> torch.Tensor:
        """

        :param x: input tensor of shape(t, b, f)
        :return:
        """
        x = x.permute(0, 2, 1)  # (b, f, t)

        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (b, 2*f, t)
        x = torch.nn.functional.glu(x, dim=1)  # (b, f, t)

        x = self.depthwise_conv(x)

        #x = x.permute(0, 2, 1)
        self.norm(x)
        x = self.activation(x)
        #x = x.permute(0, 2, 1)

        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)

        return x.permute(0, 2, 1)


class Swish(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)




        








