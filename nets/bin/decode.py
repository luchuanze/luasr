import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from nets.utils.file_utils import read_words_dict, read_non_lang_symbols
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
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--batch_size', type=int, default=4, help='batch utts for decoding')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname) %(message)s')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table, label_word = read_words_dict(args.symbol_table)

    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_conf = configs['dataset_conf']

    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    test_conf['add_noise'] = False

    test_dataset = Dataset(args.data_type,
                           args.cv_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=0
                                  )

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

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

    load_chekpoint(model, args.check_point)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model.eval()

    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for idx, batch in enumerate(test_data_loader):
            utts, feats, target, feats_lengths, target_lengths = batch


