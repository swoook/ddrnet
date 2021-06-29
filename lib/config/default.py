
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode


_C = CfgNode()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 2
_C.MODEL.EXTRA = CfgNode(new_allowed=True)


_C.MODEL.OCR = CfgNode()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

_C.LOSS = CfgNode()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]

# DATASET related params
_C.DATASET = CfgNode()
_C.DATASET.MODEL = 'train'
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

# training
_C.TRAIN = CfgNode()

_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10

_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [60, 80]
# _C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CfgNode()

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]

_C.TEST.OUTPUT_INDEX = -1

# debug
_C.DEBUG = CfgNode()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def config_sod_ddrnet_23_slim():
    config = CfgNode()

    config.GPUS = list()
    config.WORKERS = 4

    config.CUDNN = CfgNode()
    config.CUDNN.BENCHMARK = True
    config.CUDNN.DETERMINISTIC = False
    config.CUDNN.ENABLED = True

    config.MODEL = CfgNode()
    config.MODEL.NAME = 'ddrnet_23_slim'
    config.MODEL.NUM_OUTPUTS = 2
    config.MODEL.ALIGN_CORNERS = False
    config.MODEL.PRETRAINED = None
    # TODO: If it's not required, remove it
    config.MODEL.EXTRA = CfgNode(new_allowed=True)

    # TODO: If it's not required, remove it
    config.MODEL.OCR = CfgNode()
    config.MODEL.OCR.MID_CHANNELS = 512
    config.MODEL.OCR.KEY_CHANNELS = 256
    config.MODEL.OCR.DROPOUT = 0.05
    config.MODEL.OCR.SCALE = 1

    config.DATASET = CfgNode()
    config.DATASET.NAME = None
    config.DATASET.TRAIN_ROOT = None
    config.DATASET.TRAIN_SET = None
    config.DATASET.TEST_ROOT = None
    config.DATASET.TEST_SET = None
    config.DATASET.NUM_CLASSES = None


    config.TRAIN = CfgNode()

    config.TRAIN.IMAGE_SIZE = list()  # [width, height]
    config.TRAIN.BASE_SIZE = None
    config.TRAIN.FLIP = True
    config.TRAIN.MULTI_SCALE = True
    config.TRAIN.DOWNSAMPLE_RATE = 1
    config.TRAIN.SCALE_FACTOR = 16

    config.TRAIN.FREEZE_LAYERS = ''
    config.TRAIN.FREEZE_EPOCHS = -1
    config.TRAIN.NONBACKBONE_KEYWORDS = []
    config.TRAIN.NONBACKBONE_MULT = 10

    config.TRAIN.RANDOM_BRIGHTNESS = False
    config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

    config.TRAIN.LR = None
    config.TRAIN.EXTRA_LR = None

    config.TRAIN.OPTIMIZER = 'sgd'
    config.TRAIN.MOMENTUM = None
    config.TRAIN.WD = None
    config.TRAIN.NESTEROV = False
    config.TRAIN.IGNORE_LABEL = -1

    config.TRAIN.BATCH_SIZE_PER_GPU = None
    config.TRAIN.SHUFFLE = True
    config.TRAIN.PRINT_FREQ = None

    config.TRAIN.BEGIN_EPOCH = None
    config.TRAIN.END_EPOCH = None
    config.TRAIN.EXTRA_EPOCH = None

    config.TRAIN.WEIGHTS_SAVE_FREQ = None
    config.TRAIN.WEIGHTS_SAVE_DIR = None
    config.TRAIN.RESUME = None


    config.TEST = CfgNode()

    config.TEST.IMAGE_SIZE = list()  # [width, height]
    config.TEST.BASE_SIZE = None

    config.TEST.BATCH_SIZE_PER_GPU = None

    config.TEST.MODEL_FILE = None
    config.TEST.FLIP_TEST = False
    config.TEST.MULTI_SCALE = False
    config.TEST.SCALE_LIST = [1]

    config.TEST.OUTPUT_INDEX = None

    # debug
    config.DEBUG = CfgNode()
    config.DEBUG.DEBUG = False
    config.DEBUG.SAVE_BATCH_IMAGES_GT = False
    config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    config.DEBUG.SAVE_HEATMAPS_GT = False
    config.DEBUG.SAVE_HEATMAPS_PRED = False
    
    return config

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

