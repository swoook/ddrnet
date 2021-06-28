# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--runmode',
                        choices=['infer', 'fps'],
                        default='fps',
                        type=str)
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--input_size',
                        default='512',
                        help='Assume the input is a square',
                        type=int)
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    device = torch.device('cuda' if args.cuda else 'cpu')

    # cudnn related setting
    if args.cuda:
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = models.__dict__[config.MODEL.NAME]
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.SyncBatchNorm    

    model = models.__dict__[config.MODEL.NAME].__dict__['get_seg_model'](config).to(device)
    width, height = config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]
    dump_input = torch.rand((1, 3, width, height)).to(device)
    logger.info(get_model_summary(model, dump_input))

    if args.runmode == 'fps': 
        speed_test(model, size=(args.input_size, args.input_size), num_repet=1000, is_cuda=args.cuda)
    elif args.runmode == 'infer': 
        
        pass
    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     # model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
    #     model_state_file = os.path.join(final_output_dir, 'best.pth')    
    # logger.info('=> loading model from {}'.format(model_state_file))
        
    # pretrained_dict = torch.load(model_state_file)
    # if 'state_dict' in pretrained_dict:
    #     pretrained_dict = pretrained_dict['state_dict']
    # model_dict = model.state_dict()
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
    #                     if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()

    # # prepare data
    # test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # test_dataset = eval('datasets.'+config.DATASET.DATASET)(
    #                     root=config.DATASET.ROOT,
    #                     list_path=config.DATASET.TEST_SET,
    #                     num_samples=None,
    #                     num_classes=config.DATASET.NUM_CLASSES,
    #                     multi_scale=False,
    #                     flip=False,
    #                     ignore_label=config.TRAIN.IGNORE_LABEL,
    #                     base_size=config.TEST.BASE_SIZE,
    #                     crop_size=test_size,
    #                     downsample_rate=1)

    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True)

    # test(config, 
    #         test_dataset, 
    #         testloader, 
    #         model,
    #         sv_dir=final_output_dir+'/test_result')


if __name__ == '__main__':
    main()
