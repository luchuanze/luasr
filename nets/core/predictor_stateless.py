
import torch


class Predictor(torch.nn.Module):
    """
    RNN-transducer with stateless prediction network
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 blank_id: int,
                 context_size: int
                 ):
        super(Predictor, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=blank_id,
        )

        self.blank_id = blank_id
        assert context_size >= 1, context_size
        self.context_size = context_size
        self.conv = None

        if self.context_size > 1:
            self.conv = torch.nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=context_size,
                padding=0,
                groups=embedding_dim,
                bias=False,
            )

    def forward(self,
                y: torch.Tensor,
                need_pad: bool = True
                ) -> torch.Tensor:
        """

        :param y:  a tensor of shape (b, u) with blank
        :param need_pad:
        True to left pad the input. should be true during training
        False to not pad the input. should be false during inference.
        :return:
        a tensor of shape (b, u, embedding_dim)
        """

        #print(self.context_size)
        embed = self.embedding(y)
        if self.context_size > 1:
            embed = embed.permute(0, 2, 1)  # (b, dim, u)
            if need_pad is True:
                embed = torch.nn.functional.pad(
                    embed, pad=(self.context_size - 1, 0)
                )
            else:
                assert embed.size(-1) == self.context_size
            embed = self.conv(embed)
            embed = embed.permute(0, 2, 1)

        return embed
