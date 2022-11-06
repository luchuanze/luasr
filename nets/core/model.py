
import torch
from nets.core.encoder_transformer import Transformer as EncoderTransformer
from nets.core.encoder_conformer2 import Conformer as EncoderConformer
from nets.core.predictor_stateless import Predictor, PredictorLstm
from nets.core.loss_transducer import TransducerFast as LossTransducer, Joiner, JoinerFast, JoinerFast2
from nets.core.utils import load_cmvn
from nets.core.loss_transformer import Transformer as LossTransformer
from nets.core.loss_ctc import CTC as LossCTC
from nets.core.decoder_transformer2 import Transformer as DecoderTransformer


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

        self.encoder = None
        if encoder_type == 'transformer':
            self.encoder = EncoderTransformer(
                input_size=input_dim,
                global_cmvn=global_cmvn,
                output_dim=encoder_conf['output_dim'],
                attention_dim=encoder_conf['attention_dim'],
                attention_heads=encoder_conf['attention_heads'],
                feedforward_size=encoder_conf['feedforward_size'],
                num_layers=encoder_conf['num_layers'],
                normalize_before=encoder_conf['normalize_before']
            )
        elif encoder_type == 'conformer':

            self.encoder = EncoderConformer(
                input_size=input_dim,
                global_cmvn=global_cmvn,
                output_dim=encoder_conf['output_dim'],
                attention_dim=encoder_conf['attention_dim'],
                attention_heads=encoder_conf['attention_heads'],
                feedforward_size=encoder_conf['feedforward_size'],
                num_layers=encoder_conf['num_layers'],
                cnn_module_kernel=15,
                normalize_before=encoder_conf['normalize_before']
            )

        else:
            raise ModuleNotFoundError("Encoder model is not found with type of {}.".format(encoder_type))

        embedding_dim = encoder_conf['output_dim']
        predictor = Predictor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            blank_id=0,
            context_size=4
        )
        # predictor = PredictorLstm(
        #     vocab_size=vocab_size,
        #     embedding_dim=embedding_dim,
        #     blank_id=0,
        #     num_layers=4
        # )

        self.blank = predictor.blank_id

        encoder_output_dim = encoder_conf['output_dim']

        joiner = JoinerFast2(
            input_dim=encoder_output_dim,
            output_dim=vocab_size
        )

        self.transducer = LossTransducer(
            predictor=predictor,
            joiner=joiner
        )

        self.ctc = LossCTC(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_dim
        )
        # self.ctc = None
        # self.transducer = TransducerOptimized(
        #     encoder=encoder,
        #     predictor=predictor,
        #     joiner=joiner,
        #     optimized_prob=0.5
        # )

        decoder = DecoderTransformer(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_dim,
            attention_heads=4,
            feedforward_size=2048,
            num_layers=6,
        )

        self.transformer = LossTransformer(
            decoder=decoder,
            vocab_size=vocab_size,
            sos_id=4232,
            eos_id=4232,
            ignore_id=0,
            label_smoothing=0.1
        )

    def forward(self,
                x: torch.Tensor,
                x_lens: torch.Tensor,
                y: torch.Tensor,
                y_lens: torch.Tensor
                ) -> torch.Tensor:

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 2, y.shape

        assert x.size(0) == x_lens.size(0) == y.size(0)

        encoder_out, encoder_out_lens, _ = self.encoder(x, x_lens)

        assert torch.all(encoder_out_lens > 0)

        weight = 0.6
        ctc_weight = 0.2
        loss1 = 0.0
        loss2 = 0.0
        if weight > 0:
            loss_transducer = self.transducer(encoder_out, encoder_out_lens, y, y_lens)
            loss1 = loss_transducer.item()
        else:
            loss_transducer = None

        if 1 - weight > 0:
            loss_transformer, _ = self.transformer(encoder_out, encoder_out_lens, y, y_lens)
            loss2 = loss_transformer.item()
        else:
            loss_transformer = None

        loss_ctc = self.ctc(encoder_out, encoder_out_lens, y, y_lens)

        if loss_transformer is None:
            loss = loss_transducer
        elif loss_transducer is None:
            loss = loss_transformer
        else:
            loss = weight * loss_transducer + (1 - weight - ctc_weight) * loss_transformer + ctc_weight * loss_ctc
            #loss = weight * loss_transducer + (1 - weight) * loss_transformer

        #loss = loss_transformer
        return loss, loss1, loss2



