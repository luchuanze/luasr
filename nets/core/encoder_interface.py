import math
from typing import Tuple

import torch


class EncoderInterface(torch.nn.Module):
    def forward(self,
                x: torch.Tensor, x_lens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x:
        A tensor of shape(b, t, f) containing the input features.
        :param x_lens:
        A tensor of shape(b, ) containing the num  of frames in x without padding.
        :return:
        Return a tuple containing two tensors:
          -encoder_out, a tensor of shape(b, t_o, d_o)
           containing unnormalized probabilities.
          -encoder_out_lens, a tensor of shape(b, )
           containing the num of frames in encoder_out after padding.
        """
        raise NotImplementedError("Please implement it in subclass")


class Conv2dSubsampling(torch.nn.Module):
    """
    Convolutional 2D subsampling to 1/4 length.
    Convert an input of shape(b, t, f) to an output
    with shape(b, t', f'), where
    t' = ((t-1)//2 -1)//2, which approximates t' = t'//4
    """

    def __init__(self,
                 idim: int,
                 odim: int
                 ) -> None:

        assert idim >= 7
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=odim, kernel_size=3, stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=odim, out_channels=odim, kernel_size=3, stride=2
            ),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:

        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x


class PositionalEncoding(torch.nn.Module):
    """
    PE(pos, 2i) = sin(pos/ 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    1 / (10000^(2i/d_model)) = exp(-log(10000^(2i/d_model)))
                               = exp(-1* 2i / d_model * log(100000))
                               = exp(2i * -(log(10000) / d_model))
    """

    def __init__(self,
                 dim: int,
                 dropout: float = 0.1,
                 ) -> None:

        super().__init__()
        self.dim = dim
        self.scale = math.sqrt(self.dim)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.pe = torch.zeros(1, 0, self.dim, dtype=torch.float32)

    def extend_pe(self,
                  x: torch.Tensor
                  ) -> None:

        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim, dtype=torch.float32)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # pe is of shape(1, t, dim) where t is x.size(1）
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        :param x:
        Its shape is (b, t, c)
        :return:
        tensor of shape(b, t, c)
        """
        self.extend_pe(x)
        pos_emb = self.pe[:, : x.size(1)]
        x = x * self.scale + pos_emb
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):

    def __init__(self,
                 dim,
                 dropout: float = 0.1
                 ):
        super(RelPositionalEncoding, self).__init__(dim, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x)
        x = x * self.scale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:

    assert lengths.ndim == 1, lengths.ndim
    max_len = lengths.max()
    batch = lengths.size(0)
    mask = torch.arange(max_len).expand(batch, max_len).to(lengths)

    return mask >= lengths.unsqueeze(1)


