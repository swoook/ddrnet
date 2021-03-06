# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np
from torch._C import dtype
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def pixel_acc(self, pred, label):
    if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
        pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    acc  = self.pixel_acc(outputs[1], labels)
    return torch.unsqueeze(loss,0), outputs, acc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

import cv2
from PIL import Image

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

class Vedio(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 480))

    def addImage(self, img, colorMask):
        img = img[:,:,::-1]
        colorMask = colorMask[:,:,::-1]         # shape:
        img = np.concatenate([img, colorMask], axis=1)
        self.cap.write(img)

    def releaseCap(self):
        self.cap.release()


class Map16(object):
    def __init__(self, vedioCap, visualpoint=True, names=None):
        self.names = names
        if self.names is None:
            self.self.names = ("background", "floor", "bed", "cabinet,wardrobe,bookcase,shelf",
                    "person", "door", "table,desk,coffee", "chair,armchair,sofa,bench,swivel,stool",
                    "rug", "railing", "column", "refrigerator", "stairs,stairway,step", "escalator", "wall",
                    "dog", "plant")
        self.colors  = np.array([[0, 0, 0],
                    [0, 0, 255],
                    [0, 255, 0],
                    [0, 255, 255],
                    [255, 0, 0 ],
                    [255, 0, 255 ], 
                    [255, 255, 0 ],
                    [255, 255, 255 ],
                    [0, 0, 128 ],
                    [0, 128, 0 ],
                    [128, 0, 0 ],
                    [0, 128, 128 ],
                    [128, 0, 0 ],
                    [128, 0, 128 ],
                    [128, 128, 0 ],
                    [128, 128, 128 ],
                    [192, 192, 192 ]], dtype=np.uint8)
        self.outDir = "output/map16"
        self.vedioCap = vedioCap
        self.visualpoint = visualpoint
    
    def visualize_result(self, data, pred, dir, img_name=None):
        img = data

        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx]]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

        # calculate point
        if self.visualpoint:
            #???????????????float32??????????????????
            img = img.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = np.float32(img_gray)
            #????????????????????????
            goodfeatures_corners = cv2.goodFeaturesToTrack(img_gray, 400, 0.01, 10)
            goodfeatures_corners = np.int0(goodfeatures_corners)
            # ?????????????????????????????????????????????
            for i in goodfeatures_corners:
                #?????????i ?????????????????????????????????????????????flatten??????ravel?????????    
                x,y = i.flatten()
                cv2.circle(img,(x,y), 3, [0,255,], -1)

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        im_vis = img * 0.7 + pred_color * 0.3
        im_vis = im_vis.astype(np.uint8)

        # for vedio result show
        self.vedioCap.addImage(im_vis, pred_color)

        img_name = img_name
        if not os.path.exists(dir):
            os.makedirs(dir)
        Image.fromarray(im_vis).save(
            os.path.join(dir, img_name))


def letterbox_resize(src, dst_h, dst_w):
    # https://stackoverflow.com/a/48450206
    # https://github.com/qqwweee/keras-yolo3/issues/330#issue-401027524
    """Resize with same aspect ratio as source image.

    Args:
        src: `np.array`, source image.
        dst_h: `int`, height of target image.
        dst_w: `int`, width of target image.

    Returns:
        `np.array`, target image with same aspect ratio as source image.
    """
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    scale_factor = min(dst_h/src_h, dst_w/src_w)
    inter_h, inter_w = int(src_h * scale_factor), int(src_w * scale_factor)
    resized = cv2.resize(src, (inter_w, inter_h))
    top_idx = (dst_h-inter_h)//2
    left_idx = (dst_w-inter_w)//2
    dst[top_idx:top_idx+inter_h, left_idx:left_idx+inter_w, :] = resized
    return dst


def letterbox_resize_tensor(src, dst_h, dst_w):
    # https://stackoverflow.com/a/66539730
    # It seems GPU is not helpful in this case
    """Resize with same aspect ratio as source image.

    Args:
        src: `torch.tensor`, source image which has a size of (N, C, H, W).
        dst_h: `int`, height of target image.
        dst_w: `int`, width of target image.

    Returns:
        `torch.tensor`, target image with same aspect ratio as source image.
    """
    dst = torch.zeros((1, 3, dst_h, dst_w)).to(src.device)
    src_h, src_w = src.size[2:]
    scale_factor = min(dst_h/src_h, dst_w/src_w)
    inter_h, inter_w = int(src_h * scale_factor), int(src_w * scale_factor)
    resized = F.interpolate(src, (inter_w, inter_h), mode='bilinear', align_corners=False)
    top_idx = (dst_h-inter_h)//2
    left_idx = (dst_w-inter_w)//2
    dst[:, :, top_idx:top_idx+inter_h, left_idx:left_idx+inter_w] = resized
    return dst


@torch.no_grad()
def speed_test(model, size=(896, 896), num_repet=100, is_cuda=True):
    input_t = torch.Tensor(1, 3, size[0], size[1])
    if is_cuda: input_t = input_t.cuda()
    # print(next(model.parameters()).is_cuda)
    print("start warm up")
    for _ in range(30): model(input_t)
    print("warm up done")

    if is_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_repet):
        pred = model(input_t)[0]
        pred = F.interpolate(input=pred, size=size, mode='bilinear', align_corners=False)
        pred = torch.argmax(pred.exp(), dim=1)
    if is_cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()
    inference_time = (t1 - t0) / num_repet
    print('FPS: {}'.format((1/inference_time)))


@torch.no_grad()
def infer(model, input_path, output_path, input_size, device, config):
    '''
    1. Load an image
    2. Scale
    3. Normalize
    4. Infer
    5. Scale
    '''

    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    if not os.path.exists(input_path): raise FileNotFoundError
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img_shape = img.shape

    # img = letterbox_resize(img, input_size[0], input_size[1])

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = img.astype(np.float32)[:, :, ::-1]
    img = img / 255.0
    img -= mean
    img /= std
    img = img.transpose((2, 0, 1))
    img = torch.Tensor(img).to(device).unsqueeze(0)

    pred = model(img)

    if config.MODEL.NUM_OUTPUTS > 1: pred = pred[config.TEST.OUTPUT_INDEX]
    pred = torch.softmax(F.interpolate(
        input=pred, size=img_shape[:2],
        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
    ), 1).exp().argmax(dim=1)
    pred = pred.squeeze().detach().cpu().numpy()

    pred = palette[pred]
    cv2.imwrite(output_path, pred)

    return pred