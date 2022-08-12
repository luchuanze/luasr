import argparse
import copy
import logging
import os
import yaml

import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from nets.utils.file_utils import read_symbol_table, read_non_lang_symbols
from nets.dataset.dataset import Dataset
from nets.core.model import TransducerTransformer
from nets.core.executor import Executor
from nets.core.checkpoint import load_chekpoint, save_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=0,
                        type=int,
                        help='''number of total processes/gpus for
                            distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=600,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument("--enc_init_mods",
                        default="encoder.",
                        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                        help="List of encoder modules \
                            to initialize ,separated by a comma")
    parser.add_argument("--start_epoch",
                        default=0,
                        type=int,
                        help='start epoch to training')
    parser.add_argument("--num_epoch",
                        default=100,
                        type=int,
                        help='number of epoch to training')

    args = parser.parse_args()
    return args


def run(rank, args):

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info('run rank:{}'.format(rank))

    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    distributed = args.world_size > 1
    if distributed:
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=rank)

    symbol_table = read_symbol_table(args.symbol_table)

    #print(symbol_table)
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['shuffle'] = False
    cv_conf['add_noise'] = False

    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, args.bpe_model, non_lang_syms, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         symbol_table,
                         cv_conf,
                         args.bpe_model,
                         non_lang_syms,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True

    if rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Create asr model from configs
    model_conf = configs['model']
    model_conf['cmvn_file'] = args.cmvn
    model_conf['is_json_cmvn'] = True
    model = TransducerTransformer(
        input_dim=input_dim,
        vocab_size=vocab_size,
        configs=model_conf)

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info('the number of model params is {}.'.format(num_params))

    start_epoch = 0
    num_epoch = args.num_epoch
    if args.checkpoint is not None:
        checkpoint_info = load_chekpoint(model, args.checkpoint)
        start_epoch = checkpoint_info.get('epoch', -1) + 1
        cv_loss = checkpoint_info.get('cv_loss', 0.0)
        step = checkpoint_info.get('step', -1)
    else:
        checkpoint_info = {}

    #if rank == 0:
    #    script_model = torch.jit.script(model)
    #    script_model.save(os.path.join(args.model_dir, 'init.zip'))

    assert (torch.cuda.is_available())
    model.cuda()
    device = torch.device('cuda')
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )

    executor = Executor(model, device, rank,
                        accum_grad=configs['accum_grad'],
                        grad_clip=configs['grad_clip'],
                        is_dist=distributed,
                        log_interval=configs['log_interval'],
                        optimizer_conf=configs['optim_conf'],
                        scheduler_conf=configs['scheduler_conf'])

    for epoch in range(start_epoch, num_epoch):
        train_dataset.set_epoch(epoch)
        lr = executor.get_lr()
        logging.info('Epoch {} TRAIN info lr {:.8f}'.format(epoch, lr))
        executor.train(epoch, train_data_loader)

        total_loss, num_seen_utts = executor.cv(epoch, cv_data_loader)

        cv_loss = total_loss / num_seen_utts
        logging.info('Epoch {} CV loss {}.'.format(epoch, cv_loss))

        if rank == 0:
            save_path = os.path.join(args.model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model,
                save_path,
                {'epoch': epoch, 'lr': lr, 'cv_loss': cv_loss, 'step': executor.step}
            )


def main():

    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)

    logging.info('Start training.')

    world_size = args.world_size
    assert world_size >= 0
    #if world_size > 1:
    #    logging.info('training on multiple gpus')
    #    mp.spawn(run, args=(args,), nprocs=world_size, join=True)
    #else:
    #    logging.info('training on single gpus')
    #    run(rank=0, args=args)
    run(args.rank, args=args)


if __name__ == '__main__':
    main()
