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
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, letterbox_resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runmode', type=str, 
    choices=['infer', 'fps'], default='infer', 
    help='infer: infer from single image and write a result as a .jpg file \n fps: measure a FPS of given model')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--arch', type=str, 
    choices=['resnet18', 'resnet50', 'vggnet16'], 
    default='resnet', help='resnet or vgg')
    parser.add_argument('--input_img_path', metavar='PATH', help='Input image path')
    parser.add_argument('--output_img_path', metavar='PATH', required=True, 
    help='Output image path, i.e. It visualizes an inference result')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    args = parser.parse_args()
    update_config(config, args)
    return args


class Inspector():
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.net = self.load_net()
        self.net.eval()
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        

    def load_net(self,):
        # cudnn related setting
        if self.args.cuda:
            cudnn.benchmark = config.CUDNN.BENCHMARK
            cudnn.deterministic = config.CUDNN.DETERMINISTIC
            cudnn.enabled = config.CUDNN.ENABLED

        # build model
        if torch.__version__.startswith('1'):
            module = models.__dict__[config.MODEL.NAME]
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.SyncBatchNorm
        '''
        TODO: This doesn't load pre-trained weights from '.pth' file for Cityscapes appropriately
        It seems it expects '.pth' file for ImageNet
        Keys in the '.pth' file for Cityscapes are like 'model.conv1.0.weight'
        Keys in the $model_dict are like 'conv1.0.weight'
        That's why it states `if k[6:] in model_dict.keys()`
        '''
        model = module.__dict__['get_seg_model'](config, pretrained=False).to(self.device)

        # load pre-trained weights
        pretrained_dict = torch.load(self.config.MODEL.PRETRAINED, map_location=self.device)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


    @torch.no_grad()
    def infer(self):
        '''
        1. Load an image
        2. Scale
        3. Normalize
        4. Infer
        5. Scale
        '''

        if not os.path.exists(self.args.input_img_path): raise FileNotFoundError
        img = cv2.imread(self.args.input_img_path, cv2.IMREAD_COLOR)
        img_shape = img.shape

        # img = letterbox_resize(img, self.args.input_size, self.args.input_size)

        img = img.astype(np.float32)[:, :, ::-1]
        img = img / 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose((2, 0, 1))
        img = torch.Tensor(img).to(self.device).unsqueeze(0)

        pred = self.net(img)

        if config.MODEL.NUM_OUTPUTS > 1: pred = pred[config.TEST.OUTPUT_INDEX]
        pred = torch.softmax(F.interpolate(
            input=pred, size=img_shape[:2],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        ), 1).argmax(dim=1)
        pred = pred.squeeze().detach().cpu().numpy()

        pred = self.palette[pred]
        cv2.imwrite(self.args.output_img_path, pred)

        return pred


def main(args):
    inspector = Inspector(config, args)
    if args.runmode == 'infer':
        inspector.infer()
    elif args.runmode == 'fps': NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    main(args)