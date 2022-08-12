
import torch
from nets.core.encoder_transformer import Transformer as EncoderTransformer
from nets.core.predictor_stateless import Predictor
from nets.core.transducer import Transducer, TransducerOptimized, Joiner
from nets.core.utils import load_cmvn


class GlobalCMVN(torch.nn.Module):
    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


class TransducerTransformer(torch.nn.Module):

    def __init__(self, input_dim, vocab_size, configs):

        super(TransducerTransformer, self).__init__()

        if configs['cmvn_file'] is not None:
            mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(),
                torch.from_numpy(istd).float()
            )
        else:
            global_cmvn = None

        encoder_conf = configs['encoder']
        encoder_type = encoder_conf.get('type', 'transformer')

        if encoder_type == 'transformer':
            encoder = EncoderTransformer(
                input_size=input_dim,
                global_cmvn=global_cmvn,
                output_dim=encoder_conf['output_dim'],
                attention_dim=encoder_conf['attention_dim'],
                attention_heads=encoder_conf['attention_heads'],
                feedforward_size=encoder_conf['feedforward_size'],
                num_layers=encoder_conf['num_layers'],
                normalize_before=encoder_conf['normalize_before']
            )
        else:
            raise ModuleNotFoundError("Encoder model is not found with type of {}.".format(encoder_type))

        embedding_dim = encoder_conf['output_dim']
        predictor = Predictor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            blank_id=0,
            context_size=2
        )

        encoder_output_dim = encoder_conf['output_dim']

        joiner = Joiner(
            input_dim=encoder_output_dim,
            output_dim=vocab_size
        )

        self.transducer = Transducer(
            encoder=encoder,
            predictor=predictor,
            joiner=joiner
        )
        # self.transducer = TransducerOptimized(
        #     encoder=encoder,
        #     predictor=predictor,
        #     joiner=joiner,
        #     optimized_prob=0.5
        # )

    def forward(self,
                x: torch.Tensor,
                x_lens: torch.Tensor,
                y: torch.Tensor,
                y_lens: torch.Tensor
                ) -> torch.Tensor:

        loss = self.transducer(x, x_lens, y, y_lens)

        return loss



