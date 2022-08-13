import os.path
import re

import torch
import logging

import yaml


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):

    logging.info('Checkpoint: save to checkpoint {}.'.format(path))
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def load_chekpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from chechpoint {} for GPU.'.format(path))
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint {} for CPU.'.format(path))
        checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs