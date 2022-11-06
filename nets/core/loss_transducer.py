import random

import torch
from nets.core.encoder_interface import EncoderInterface
import torchaudio
import fast_rnnt


class Transducer(torch.nn.Module):

    def __init__(self,
                 predictor: torch.nn.Module,
                 joiner: torch.nn.Module,
                 optimized_prob: float = 0.0
                 ):
        super(Transducer, self).__init__()
        assert hasattr(predictor, "blank_id")
        self.predictor = predictor
        self.joiner = joiner
        self.optimized_prob = optimized_prob

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_out_lens: torch.Tensor,
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

        blank_id = self.predictor.blank_id

        sos_y_pad = torch.nn.functional.pad(y, pad=(1, 0, 0, 0), value=0.0)
        sos_y_pad = sos_y_pad.to(torch.int64)

        predictor_out = self.predictor(sos_y_pad)

        logits = self.joiner(encoder_out, predictor_out)

        y_padded = y.to(torch.int32)

        loss = self.compute_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=encoder_out_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="mean",
        )

        return loss

    def compute_loss(self,
                     logits,
                     targets,
                     logit_lengths,
                     target_lengths,
                     blank,
                     reduction):
        return torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=blank,
            reduction=reduction,
        )


class TransducerFast(Transducer):

    def __init__(self,
                 predictor: torch.nn.Module,
                 joiner: torch.nn.Module,
                 optimized_prob: float = 0.0
                 ):
        super(TransducerFast, self).__init__(
            predictor,
            joiner,
            optimized_prob
        )

        self.simple_am_proj = ScaledLinear(
            256,
            4233,
            initial_speed=0.5
        )
        self.simple_lm_proj = ScaledLinear(
            256,
            4233
        )

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_out_lens: torch.Tensor,
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

        blank_id = self.predictor.blank_id
        batch_size = encoder_out.size(0)

        sos_y_pad = torch.nn.functional.pad(y, pad=(1, 0, 0, 0), value=0.0)
        sos_y_pad = sos_y_pad.to(torch.int64)

        predictor_out = self.predictor(sos_y_pad)  # (b, l+1, f)

        symbols = y.to(torch.int64)

        boundary = torch.zeros((batch_size, 4), dtype=torch.int64, device=encoder_out.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        am = self.simple_am_proj(encoder_out)
        lm = self.simple_lm_proj(predictor_out)

        simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_simple(
            lm=lm,
            am=am,
            symbols=symbols,
            termination_symbol=blank_id,
            boundary=boundary,
            reduction="mean",
            return_grad=True,
        )

        prune_range = 5
        ranges = fast_rnnt.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )
        # ranges (b, t, s_ranges)

        am_pruned, lm_pruned = fast_rnnt.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.predictor_proj(predictor_out),
            ranges=ranges
        )

        logits = self.joiner(am_pruned, lm_pruned)

        pruned_loss = fast_rnnt.rnnt_loss_pruned(
            logits=logits,
            symbols=symbols,
            ranges=ranges,
            termination_symbol=blank_id,
            boundary=boundary,
            reduction="mean",
        )

        return pruned_loss


class ScaledLinear(torch.nn.Linear):
    """
    A modified version of nn.Linear where the parameters are scaled before
    use, via:
         weight = self.weight * self.weight_scale.exp()
         bias = self.bias * self.bias_scale.exp()

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
        initial_speed: this affects how fast the parameter will
           learn near the start of training; you can set it to a
           value less than one if you suspect that a module
           is contributing to instability near the start of training.
           Nnote: regardless of the use of this option, it's best to
           use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.   Must be greater than 0.
    """

    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs
    ):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = torch.nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = torch.nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)

        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in nn.Linear

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3 ** 0.5) * std
        torch.nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        if self.bias is None or self.bias_scale is None:
            return None
        else:
            return self.bias * self.bias_scale.exp()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            input, self.get_weight(), self.get_bias()
        )


class TransducerOptimized(Transducer):

    def __init__(self,
                 encoder: EncoderInterface,
                 predictor: torch.nn.Module,
                 joiner: torch.nn.Module,
                 optimized_prob: float = 0.0
                 ):
        super(TransducerOptimized, self).__init__(encoder,
                                                  predictor,
                                                  joiner,
                                                  optimized_prob)

    def compute_loss(self,
                     logits,
                     targets,
                     logit_lengths,
                     target_lengths,
                     blank,
                     reduction):

        assert 0.0 <= self.optimized_prob <= 1, self.optimized_prob

        import optimized_transducer

        if self.optimized_prob == 0:
            one_sym_per_frame = False
        elif random.random() < self.optimized_prob:
            one_sym_per_frame = True
        else:
            one_sym_per_frame = False

        return optimized_transducer.transducer_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=blank,
            reduction="mean",
            one_sym_per_frame=one_sym_per_frame,
            from_log_softmax=False
        )


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


class JoinerOptimized(Joiner):
    def __init__(self,
                 input_dim: int,
                 output_dim: int
                 ):
        super(JoinerOptimized, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim
        )
        #self.output_linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self,
                encoder_out: torch.Tensor,
                predictor_out: torch.Tensor,
                encoder_out_len: torch.Tensor,
                predictor_out_len: torch.Tensor
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


class JoinerFast(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 inner_dim: int=512
                 ):

        super(JoinerFast, self).__init__()
        self.inner_linear = torch.nn.Linear(input_dim, inner_dim)
        self.output_linear = torch.nn.Linear(inner_dim, output_dim)
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self,
                encoder_out: torch.Tensor,
                predictor_out: torch.Tensor,
                ) -> torch.Tensor:

        assert encoder_out.ndim == predictor_out.ndim
        assert encoder_out.ndim in (2, 4)
        assert encoder_out.shape == predictor_out.shape

        logit = encoder_out + predictor_out
        logit = self.inner_linear(torch.tanh(logit))
        output = self.output_linear(torch.nn.functional.relu(logit))

        return output


class JoinerFast2(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 inner_dim: int=512
                 ):

        super(JoinerFast2, self).__init__()
        self.encoder_proj = ScaledLinear(input_dim, inner_dim)
        self.predictor_proj = ScaledLinear(input_dim, inner_dim)
        self.output_linear = ScaledLinear(inner_dim, output_dim)
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self,
                encoder_out: torch.Tensor,
                predictor_out: torch.Tensor,
                ) -> torch.Tensor:

        assert encoder_out.ndim == predictor_out.ndim
        assert encoder_out.ndim in (2, 4)
        assert encoder_out.shape == predictor_out.shape

        logit = encoder_out + predictor_out
        output = self.output_linear(torch.tanh(logit))

        return output