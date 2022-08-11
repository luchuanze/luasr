import argparse
import os.path
import copy
import yaml
import logging
import torch.multiprocessing as mp
from nets.utils.file_utils import AttrDict
from nets.dataset.dataset import Dataset
from torch.utils.data import DataLoader
import nets.transducer.executor as Transducer


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


def run(rank, args):

    configfile = open(args.config)
    configs = yaml.load(configfile, Loader=yaml.FullLoader)
    config = AttrDict(configs)

    exp_name = 'exp'
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['shuffle'] = False
    non_lang_syms = None

    symbol_table = read_symbol_table(args.symbol_table)

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

    model_config = AttrDict(configs['model'])
    optim_config = AttrDict(configs['optim'])

    vocab_size = len(symbol_table)
    model_config.vocab_size = vocab_size

    if model_config.type == 'transducer':
        logging.info("training transducer")
        Transducer.train(start_epoch=args.start_epoch,
                         epochs=config.training.epochs,
                         train_data_loader=train_data_loader,
                         dev_data_loader=cv_data_loader,
                         rank=rank,
                         world_size=args.world_size,
                         model_params=model_config,
                         optim_params=optim_config)
    else:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='num of GPUs for DDP training')
    parser.add_argument("--start-epoch",
                        type=int,
                        default=0,
                        help='resume training from from this epoch')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')

    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=args, nprocs=world_size, join=True)
    else:
        run(rank=0, args=args)


if __name__ == '__main__':
    main()
