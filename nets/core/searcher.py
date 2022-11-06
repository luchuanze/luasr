import numpy as np
import torch.nn
from typing import Dict, List, Optional


class Searcher:
    def __init__(self,
                 model: torch.nn.Module,
                 device
                 ):
        self.model = model
        self.device = device

    def run(self,
            feats: torch.Tensor,
            feat_lens: torch.Tensor
            ) -> List[int]:

        feats = feats.to(self.device)
        feats_lens = feat_lens.to(self.device)

        encoder_out, encoder_out_lens, mask = self.model.encoder(
            feats, feats_lens
        )

        # hyp = transducer_greedy(model=self.model.transducer,
        #                       encoder_out=encoder_out,
        #                       device=self.device)
        # hyp, _ = ctc_greedy(model=self.model,
        #                          encoder_out=encoder_out,
        #                          device=self.device)

        attention_all(model=self.model,
                      encoder_out=encoder_out,
                      encoder_out_mask=mask,
                      device=self.device)

        return hyp


def transducer_greedy(
        model: torch.nn.Module,
        encoder_out: torch.Tensor,
        device,
) -> List[int]:
    """

    :param model: An model with transducer loss
    :param encoder_out: A tensor of shape (b, t, c)
    :return: search result list
    """
    assert encoder_out.ndim == 3
    # support only one utts for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank = model.predictor.blank_id
    context_size = model.predictor.context_size
    #context_size = 1

    predictor_input = torch.tensor(
        [blank] * context_size, device=device
    ).reshape(1, context_size)

    predictor_output = model.predictor(predictor_input, need_pad=False)
    #predictor_output = model.predictor(predictor_input)
    T = encoder_out.size(1)
    t = 0
    hyp = [blank] * context_size

    max_sym_per_frame = 5
    max_sym_per_utt = 1000
    sym_per_frame = 0

    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue

        current_encoder_out = encoder_out[:, t:t+1, :]

        logits = model.joiner(current_encoder_out, predictor_output)
        # logits is (1, 1, 1, vocab_size)

        y = logits.argmax().item()
        if y != blank:
            hyp.append(y)
            predictor_input = torch.tensor(
                [hyp[-context_size:]], device=device
            ).reshape(1, context_size)

            predictor_output = model.predictor(predictor_input, need_pad=False)
            #predictor_output = model.predictor(predictor_input)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1

    hyp = hyp[context_size:] # remove blanks

    return hyp


class Hypothesis:
    def __init__(self, ys: List[int], log_prob: float):
        self.ys = ys
        self.log_prob = log_prob

    @property
    def key(self) -> str:
        return "_".join(map(str, self.ys))


class HypList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None):
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self):
        return self.data

    def add(self, hyp: Hypothesis):
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]
            old_hyp.log_prob = np.logaddexp(old_hyp.log_prob, hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        if length_norm:
            return max(
                self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys)
            )
        else:
            return max(
                self._data.values(), key=lambda hyp: hyp.log_prob
            )

    def remove(self, hyp: Hypothesis) -> None:
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: float) -> "HypList":
        ans = HypList()
        for key, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)
        return ans

    def nbest(self, n: int) -> "HypList":
        hyps = list(self._data.items())
        hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:n]

        ans = HypList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


def transducer_beam(
        model: torch.nn.Module,
        encoder_out: torch.Tensor,
        device,
        beam: int = 4,
) -> List[int]:
    """
    It implements Algorithm 1 in https://arxiv.org/pdf/1211.3711.pdf
    :param model:
    :param encoder_out:
    :param device:
    :param beam:
    :return:
    """

    assert encoder_out.ndim == 3

    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank = model.predictor.blank_id
    context_size = model.predictor.context_size

    predictor_input = torch.tensor(
        [blank] * context_size, device=device
    ).reshape(1, context_size)

    predictor_output = model.predictor(predictor_input, need_pad=False)

    T = encoder_out.size(1)
    t = 0

    B = HypList()
    B.add(Hypothesis(ys=[blank] * context_size, log_prob=0.0))

    max_sym_per_utt = 20000

    sym_per_utt = 0

    decode_cache: Dict[str, torch.Tensor] = {}

    while t < T and sym_per_utt < max_sym_per_utt:

        current_encoder_out = encoder_out[:, t:t+1, :]
        A = B
        B = HypList()

        joint_cache: Dict[str, torch.Tensor] = {}

        while True:
            y_start = A.get_most_probable()
            A.remove(y_start)

            cached_key = y_start.key
            if cached_key not in decode_cache:
                input = [y_start.ys[-context_size:]]
                predictor_input = torch.tensor(
                    input, device=device
                ).reshape(1, context_size)

                predictor_output = model.predictor(predictor_input, need_pad=False)
                decode_cache[cached_key] = predictor_output
            else:
                predictor_output = decode_cache[cached_key]

            cached_key += f"-t-{t}"
            if cached_key not in joint_cache:
                logits = model.joiner(current_encoder_out, predictor_output)
                log_prob = logits.log_softmax(dim=-1)
                log_prob = log_prob.squeeze()
                joint_cache[cached_key] = log_prob
            else:
                log_prob = joint_cache[cached_key]

            skip_log_prob = log_prob[blank]
            new_y_star_log_prob = y_start.log_prob + skip_log_prob.item()

            B.add(Hypothesis(ys=y_start.ys[:], log_prob=new_y_star_log_prob))

            values, indices = log_prob.topk(beam + 1)
            for i, v in zip(indices.tolist(), values.tolist()):
                if i == blank:
                    continue
                new_ys = y_start.ys + [i]
                new_log_prob = y_start.log_prob + v
                A.add(Hypothesis(ys=new_ys, log_prob=new_log_prob))

            A_most_probable = A.get_most_probable()

            kept_B = B.filter(A_most_probable.log_prob)
            if len(kept_B) >= beam:
                B = kept_B.nbest(beam)
                break
        t += 1

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]
    return ys


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1

    return new_hyp


def ctc_greedy(
        model: torch.nn.Module,
        encoder_out: torch.Tensor,
        device,
) -> List[int]:

    assert encoder_out.ndim == 3

    assert encoder_out.size(0) == 1, encoder_out.size(0)

    batch_size = encoder_out.size(0)
    max_len = encoder_out.size(1)
    ctc_probs = model.ctc.log_softmax(encoder_out=encoder_out)
    topk_prob, topk_idx = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_idx = topk_idx.view(batch_size, max_len)

    hyps = [hyp.tolist() for hyp in topk_idx]
    scores = topk_prob.max(1)
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]

    return hyps[0], scores


def attention_all(
        model: torch.nn.Module,
        encoder_out: torch.Tensor,
        encoder_out_mask: torch.Tensor,
        device,
        beam: int = 4,
):
    assert encoder_out.ndim == 3

    assert encoder_out.size(0) == 1, encoder_out.size(0)

    batch_size = encoder_out.size(0)
    max_len = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)

    running_size = batch_size * beam
    encoder_out = encoder_out.unsqueeze(1).repeat(1, beam, 1, 1).view(
        running_size, max_len, encoder_dim
    )

    encoder_out_mask = encoder_out_mask.unsqueeze(1).repeat(1, beam, 1, 1).view(
        running_size, max_len
    )

    hyps = torch.ones([running_size, 1], dtype=torch.long,
                      device=device).fill_(model.transformer.sos_id)

    scores = torch.tensor([0.0] + [-float('inf')] * (beam - 1),
                          dtype=torch.float)

    scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)

    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)

    def subsequent_mask(size: int,
                        device: torch.device = torch.device("cpu")):

        arange = torch.arange(size, device=device)
        mask = arange.expand(size, size)
        arange = arange.unsqueeze(-1)
        mask = mask <= arange
        return mask

    def mask_finished_scores(score: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:

        beam_size = score.size(-1)
        zero_mask = torch.zeros_like(flag, dtype=torch.bool)
        if beam_size > 1:
            unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])), dim=1)
            finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])), dim=1)
        else:
            unfinished = zero_mask
            finished = flag

        score.masked_fill_(unfinished, -float('inf'))
        score.masked_fill_(finished, 0)

        return score

    def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                            eos: int) -> torch.Tensor:

        beam_size = pred.size(-1)
        finished = flag.repeat([1, beam_size])
        return pred.masked_fill_(finished, eos)

    for i in range(1, max_len + 1):
        if end_flag.sum() == running_size:
            break

        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1).to(device)

        logp = model.transformer.decoder.forward_one_step(
            encoder_out=encoder_out,
            encoder_out_mask=encoder_out_mask,
            y=hyps,
            y_mask=hyps_mask)

        top_k_logp, top_k_idx = logp.topk(beam)
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        top_k_idx = mask_finished_preds(top_k_idx, end_flag, model.transformer.eos_id)

        scores = scores + top_k_logp
        scores = scores.view(batch_size, beam * beam)
        scores, offset_k_idx = scores.topk(k=beam)
        scores = scores.view(-1, 1)

        base_k_idx = torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam])
        base_k_idx = base_k_idx * beam * beam
        best_k_idx = base_k_idx.view(-1) + offset_k_idx.view(-1)

        # 2.5 Update best hyps
        best_k_pred = torch.index_select(top_k_idx.view(-1),
                                         dim=-1,
                                         index=best_k_idx)  # (B*N)
        best_hyps_index = best_k_idx // beam
        last_best_k_hyps = torch.index_select(
            hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                         dim=1)  # (B*N, i+1)

        # 2.6 Update end flag
        end_flag = torch.eq(hyps[:, -1], model.transformer.eos_id).view(-1, 1)

    # 3. Select best of best
    scores = scores.view(batch_size, beam)
    # TODO: length normalization
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, 1:]
    return best_hyps, best_scores













