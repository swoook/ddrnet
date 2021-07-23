# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
from genericpath import isfile
import os

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import sys
print(sys.path)
sys.path.append('.')
from sod.models.ddrnet_23_slim import get_seg_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runmode', type=str, 
    choices=['infer', 'fps'], default='infer', 
    help='infer: infer from single image and write a result as a .jpg file \n fps: measure a FPS of given model')
    parser.add_argument('--pretrained_path', type=str, metavar='PATH',
    help='path of model file such as .pth', default=None) # Snapshot

    parser.add_argument('--input_img_path', metavar='PATH', help='Input image path', 
    default=None)
    parser.add_argument('--output_img_path', metavar='PATH', 
    help='Output image path, i.e. It visualizes an inference result', default=None)

    parser.add_argument('--input_imgs_dir', metavar='DIR', help='directory of input images', 
    default=None)
    parser.add_argument('--output_imgs_dir', metavar='DIR', help='directory of output images', 
    default=None)

    parser.add_argument('--cpu', dest='cuda', action='store_false')
    args = parser.parse_args()
    return args


class Inspector():
    def __init__(self, args):
        self.args = args
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.load_net()
        

    def load_net(self,):
        self.net = get_seg_model(self.args, is_train=False, num_classes=1).to(self.device)
        trained_params = torch.load(self.args.pretrained_path, map_location=self.device)
        self.net.load_state_dict(trained_params)
        self.net.eval()


    @torch.no_grad()
    def infer_single_img(self, input_path, output_path):
        if not os.path.exists(input_path): raise FileNotFoundError
        try:
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        except:
            return -1

        img = img.astype(np.float32)[:, :, ::-1]
        img = img / 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose((2, 0, 1))
        img = torch.Tensor(img).to(self.device).unsqueeze(0)

        preds = self.net(img)
        pred = preds[0] if len(preds) > 1 else preds
        pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy())
        pred = 255 * pred
        output_img_dir = os.path.dirname(output_path)
        if not os.path.exists(output_img_dir): os.makedirs(output_img_dir)
        cv2.imwrite(output_path, pred)

        return pred


    def infer(self):
        '''
        1. Load an image
        2. Scale
        3. Normalize
        4. Infer
        5. Scale
        '''
        if (self.args.input_imgs_dir is None) and (self.args.output_imgs_dir is None):
            self.infer_single_img(self.args.input_img_path, self.args.output_img_path)
        elif (self.args.input_img_path is None) and (self.args.output_img_path is None):
            filenames = os.listdir(self.args.input_imgs_dir)
            input_paths = [os.path.join(self.args.input_imgs_dir, filename) 
            for filename in filenames]
            for input_path in input_paths:
                input_name = os.path.basename(input_path)
                output_path = os.path.join(self.args.output_imgs_dir, input_name)
                self.infer_single_img(input_path, output_path)


def main(args):
    inspector = Inspector(args)
    if args.runmode == 'infer':
        inspector.infer()
    elif args.runmode == 'fps': NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    main(args)