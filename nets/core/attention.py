
from typing import Optional, Tuple
import math
import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 nhead: int,
                 dropout_rate: float = 0.0):

        super(MultiHeadAttention, self).__init__()
        assert embed_dim % nhead == 0
        self.d_k = embed_dim // nhead
        self.nhead = nhead
        self.linear_q = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_k = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_v = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_out = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, -1, self.nhead, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.nhead, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.nhead, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (b, h, t, d_k)

        return q, k, v

    def forward_attention(self,
                          value: torch.Tensor,
                          scores: torch.Tensor,
                          key_padding_mask: Optional[torch.Tensor] = None,
                          attention_mask: Optional[torch.Tensor] = None,
                          ) -> torch.Tensor:

        batch_size = value.size(0)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).eq(0)  # （b, 1, *, t2）
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)   # (b, h, t1, t2)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  #(b, h, t1, d_k)
        x = (x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k))
        # now x is a tensor of shape (b, t1, embed_dim)
        return self.linear_out(x)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v,
                                      scores,
                                      key_padding_mask=key_padding_mask,
                                      attention_mask=attention_mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):

    def __init__(self,
                 embed_dim: int,
                 nhead: int,
                 dropout_rate: float = 0.0
                 ):

        super(RelPositionMultiHeadAttention, self).__init__(
            embed_dim,
            nhead,
            dropout_rate)

        self.linear_pos = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.pos_bias_u = torch.nn.Parameter(torch.Tensor(nhead, self.d_k))
        self.pos_bias_v = torch.nn.Parameter(torch.Tensor(nhead, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor
                ):

        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (b, t1, h, d_k)

        batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(batch_pos, -1, self.nhead, self.d_k)
        p = p.transpose(1, 2)  # (b, h, t1, d_k)

        q_u = (q + self.pos_bias_u).transpose(1, 2)  # (b, h, t1, d_k)
        q_v = (q + self.pos_bias_v).transpose(1, 2)  # (b, h, t1, d_k)

        mat_ac = torch.matmul(q_u, k.transpose(-2, -1))  # (b, h, t1, t2)
        mat_bd = torch.matmul(q_v, p.transpose(-2, -1))  # (b, h, t1, t2)

        scores = (mat_ac + mat_bd) / math.sqrt(self.d_k)  # (b, h, t1, t2)

        return self.forward_attention(v,
                                      scores,
                                      key_padding_mask=key_padding_mask,
                                      attention_mask=attention_mask)


class RelPositionMultiHeadAttentionX(torch.nn.Module):
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
        assert n == time1
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
