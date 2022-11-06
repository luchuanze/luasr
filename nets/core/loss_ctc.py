import torch.nn
from typeguard import check_argument_types


class CTC(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 dropout_rate: float = 0.1
                 ):

        assert check_argument_types()
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.linaer = torch.nn.Linear(encoder_output_size, vocab_size)

        self.loss = torch.nn.CTCLoss(reduction="sum")

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_out_lens: torch.Tensor,
                y: torch.Tensor,
                y_lens: torch.Tensor) -> torch.Tensor:

        logits = self.linaer(torch.nn.functional.dropout(encoder_out, p=self.dropout_rate))

        logits = logits.transpose(0, 1)
        logits = logits.log_softmax(2)
        loss = self.loss(logits, y, encoder_out_lens, y_lens)

        loss = loss / logits.size(1)
        return loss

    def log_softmax(self, encoder_out: torch.Tensor) -> torch.Tensor:

        return torch.nn.functional.log_softmax(self.linaer(encoder_out), dim=2)



