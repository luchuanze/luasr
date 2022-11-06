
# Copyright 2022 Chuanze Lu

import torch.nn
from nets.core.decoder_transformer import Transformer as DecoderTransformer
from nets.core.encoder_interface import make_pad_mask
from torch.nn.utils.rnn import pad_sequence


class Transformer(torch.nn.Module):
    def __init__(self,
                 decoder: DecoderTransformer,
                 vocab_size: int,
                 sos_id: int,
                 eos_id: int,
                 ignore_id: int,
                 label_smoothing: float
                 ):
        super(Transformer, self).__init__()

        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.ignore_id = ignore_id
        self.new_ignore_id = -1
        self.criterion = LabelSmoothingLoss(
            vocab_size,
            self.new_ignore_id,
            label_smoothing,
            normalize_length=False)

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_out_lens: torch.Tensor,
                y: torch.Tensor,
                y_lens: torch.Tensor
                ):

        encoder_out_mask = ~make_pad_mask(encoder_out_lens).unsqueeze(1)

        y_in_pad = add_sos(
            y,
            sos_id=self.sos_id,
            ignore_id=self.ignore_id,
            new_ignore_id=self.eos_id
        )
        y_out_pad = add_eos(
            y,
            eos_id=self.eos_id,
            ignore_id=self.ignore_id,
            new_ignore_id=self.new_ignore_id
        )
        y_pad_lens = y_lens + 1

        decoder_out = self.decoder(encoder_out, encoder_out_mask, y_in_pad, y_pad_lens)

        loss = self.criterion(decoder_out, y_out_pad)

        accuracy = th_accuracy(
            decoder_out.reshape(-1, self.criterion.vocab_size),
            y_out_pad,
            ignore_label=self.new_ignore_id
        )

        return loss, accuracy

    def forward_one_step(self,
                         encoder_out: torch.Tensor,
                         encoder_out_lens: torch.Tensor,
                         y: torch.Tensor,
                         y_mask: torch.Tensor
                         ):
        encoder_out_mask = make_pad_mask(encoder_out_lens)

        self.decoder.forward_one_step(encoder_out,
                                      encoder_out_mask,
                                      y,)


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 padding_id: int,
                 smoothing: float,
                 normalize_length: bool = False):

        super(LabelSmoothingLoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='none')
        self.padding_id = padding_id
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """

        :param x: prediction (b, u, c)
        :param target: target masked with self.padding_id (b, u)
        :return: the KL loss
        """

        assert x.ndim == 3
        assert target.ndim == 2
        assert x.shape[:2] == target.shape
        assert self.vocab_size == x.size(-1)

        batch_size = x.size(0)
        x = x.reshape(-1, self.vocab_size)
        # x is of shape (b*u, c)

        # do not change target in-place and make a copy of it here
        target = target.clone().reshape(-1)

        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing/(self.vocab_size - 1))
        ignore = target == self.padding_id
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)

        den = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / den


def add_sos(tgt: torch.Tensor, sos_id: int, ignore_id: int, new_ignore_id: int):

    sos = torch.tensor([sos_id], dtype=torch.long,
                        requires_grad=False,
                        device=tgt.device)

    ys = [y[y != ignore_id] for y in tgt]
    ys_out = [torch.cat([sos, y], dim=0) for y in ys]

    ys_out = pad_sequence(ys_out, batch_first=True, padding_value=new_ignore_id)

    return ys_out


def add_eos(tgt: torch.Tensor, eos_id: int, ignore_id: int, new_ignore_id: int):
    """

    :param tgt:
    :param eos_id:
    :param ignore_id:
    :param new_ignore_id:
    :return:

    x = tensor([[1, 2, 3, 4, 5],
        [4, 5, 6, 0, 0],
        [7, 9, 9, 8, 0]], dtype=torch.int32)

    y = add_eos(x, 10, 0, -1)

    y == tensor([[ 1,  2,  3,  4,  5, 10],
        [ 4,  5,  6, 10, -1, -1],
        [ 7,  9,  9,  8, 10, -1]])

    """

    eos = torch.tensor([eos_id], dtype=torch.long,
                       requires_grad=False,
                       device=tgt.device)

    ys = [y[y != ignore_id] for y in tgt]
    ys_out = [torch.cat([y, eos], dim=0) for y in ys]

    ys_out = pad_sequence(ys_out, batch_first=True, padding_value=new_ignore_id)

    return ys_out


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> float:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)






