import logging
from contextlib import nullcontext
import torch.nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int = 25000,
                 last_epoch: int = -1):

        self.warmup_steps = warmup_steps
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step


class Executor:
    def __init__(self,
                 model: torch.nn.Module,
                 device,
                 rank,
                 accum_grad: int = 1,
                 grad_clip: float = 50.0,
                 is_dist: bool = True,
                 log_interval: int = 100,
                 optimizer_conf=None,
                 scheduler_conf=None
                 ):
        self.step = 0
        self.clip = grad_clip
        self.rank = rank
        self.accum_grad = accum_grad
        self.is_distributed = is_dist
        logging.info('using accumlate grad, new batch size is {} time'
                     ' larger than before'.format(self.accum_grad))

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), **optimizer_conf)
        self.scheduler = WarmupLR(optimizer=self.optimizer, **scheduler_conf)
        self.device = device
        self.log_interval = log_interval

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, epoch, data_loader):

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_context = self.model.join
            logging.info('model is distributedDataParallel.')
        else:
            model_context = nullcontext

        self.model.train()
        with model_context():
            for idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(self.device)
                target = target.to(self.device)
                feats_lengths = feats_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue

                if self.is_distributed and idx % self.accum_grad != 0:
                    context = self.model.no_sync
                else:
                    context = nullcontext

                with context():
                    loss = self.model(feats, feats_lengths, target, target_lengths)
                    loss = loss / self.accum_grad
                    assert loss.requires_grad == True, loss.requires_grad
                    loss.backward()

                if idx % self.accum_grad == 0:
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.clip)
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                if idx % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_str = 'train batch {}/{} loss {:.6f}'.format(
                        epoch, idx,
                        loss.item() * self.accum_grad
                    )
                    log_str += ' lr {:.8f} rank {}'.format(lr, self.rank)
                    logging.debug(log_str)

    def cv(self, epoch, data_loader):

        self.model.eval()
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(self.device)
                target = target.to(self.device)
                feats_lengths = feats_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                num_utts = target_lengths.size(0)

                if num_utts == 0:
                    continue
                loss = self.model(feats, feats_lengths, target, target_lengths)

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                else:
                    logging.error('CV Batch {}/{} loss is not finite.'.format(epoch, idx))

                if idx % self.log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(epoch, idx, loss.item())
                    log_str += 'history loss {:.6f}'.format(total_loss / num_seen_utts)
                    log_str += ' rank {}'.format(self.rank)
                    logging.debug(log_str)

        return total_loss, num_seen_utts






