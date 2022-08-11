
import torch
from nets.core.encoder_interface import EncoderInterface
import torchaudio


class Transducer(torch.nn.Module):

    def __init__(self,
                 encoder: EncoderInterface,
                 predictor: torch.nn.Module,
                 joiner: torch.nn.Module,
                 ):
        super(Transducer, self).__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(predictor, "blank_id")

        self.encoder = encoder
        self.predictor = predictor
        self.joiner = joiner

    def forward(self,
                x: torch.Tensor,
                x_lens: torch.Tensor,
                y: torch.Tensor,
                y_lens: torch.Tensor
                ) -> torch.Tensor:

        """

        :param x:  shape （b, t, f）
        :param x_lens: shape (b,)
        :param y:  shape (b, l)
        :param y_lens: (b, )
        :return:
        """

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 2, y.shape

        assert x.size(0) == x_lens.size(0) == y.size(0)

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        blank_id = self.predictor.blank_id

        sos_y_pad = torch.nn.functional.pad(y, pad=(1, 0, 0, 0), value=0.0)
        sos_y_pad = sos_y_pad.to(torch.int64)

        predictor_out = self.predictor(sos_y_pad)

        logits = self.joiner(encoder_out, predictor_out)

        y_padded = y.to(torch.int32)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="mean",
        )

        return loss


class Joiner(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int
                 ):
        super(Joiner, self).__init__()
        self.output_linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self,
                encoder_out: torch.Tensor,
                predictor_out: torch.Tensor
                ) -> torch.Tensor:
        """

        :param encoder_out:
        Output from the encoder, a tensor of shape （b, t, c）
        :param predictor_out:
        Output from the predictor, a tensor of shape (b, u, c)
        :return:
        a tensor of shape （b, t, u, c）
        """
        assert encoder_out.ndim == predictor_out.ndim == 3
        assert encoder_out.size(0) == predictor_out.size(0)
        assert encoder_out.size(2) == predictor_out.size(2)

        encoder_out = encoder_out.unsqueeze(2)  # (b, t, 1, c)
        predictor_out = predictor_out.unsqueeze(1)  # (b, 1, u, c)

        logit = encoder_out + predictor_out
        logit = torch.tanh(logit)

        output = self.output_linear(logit)

        return output

