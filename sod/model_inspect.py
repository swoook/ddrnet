# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sod.models.ddrnet_23_slim import get_seg_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runmode', type=str, 
    choices=['infer', 'fps'], default='infer', 
    help='infer: infer from single image and write a result as a .jpg file \n fps: measure a FPS of given model')
    parser.add_argument('--model_path', type=str, metavar='PATH',
    help='path of model file such as .pth', default=None) # Snapshot

    parser.add_argument('--input_img_path', metavar='PATH', help='Input image path')
    parser.add_argument('--output_img_path', metavar='PATH', required=True, 
    help='Output image path, i.e. It visualizes an inference result')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    args = parser.parse_args()
    return args


class Inspector():
    def __init__(self, args):
        self.args = args
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.net = self.load_net()
        self.palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        

    def load_net(self,):
        self.net = get_seg_model(self.config, pretrained=False, num_classes=1).to(self.device)
        trained_params = torch.load(self.args.model_path, map_location=self.device)
        self.net.load_state_dict(trained_params, map_location=self.device)
        self.net.eval()


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
    inspector = Inspector(args)
    if args.runmode == 'infer':
        inspector.infer()
    elif args.runmode == 'fps': NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    main(args)