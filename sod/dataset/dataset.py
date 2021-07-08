import os
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import random
from math import ceil, floor


class DutsTrSet(data.Dataset):
    ''' 
    DUTS-TR
    '''
    def __init__(self, root_dir, list_txt_path):
        self.root_dir = root_dir
        self.path_img_gt_pairs = list_txt_path

        with open(self.path_img_gt_pairs, 'r') as f:
            self.img_gt_pairs = [x.strip() for x in f.readlines()]
        # TODO: pre-process
        self.num_data = len(self.img_gt_pairs)


    def __getitem__(self, item):
        # TODO: Is `item % self.num_data` necessary?
        img_path, label_path = self.img_gt_pairs[item % self.num_data].split()
        img = load_train_img(os.path.join(self.root_dir, img_path))
        label = load_label(os.path.join(self.root_dir, label_path))

        img, label = cv_random_flip(img, label)
        img = torch.Tensor(img)
        label = torch.Tensor(label)

        return {'img': img, 'label': label}

    def __len__(self):
        return self.num_data


class DutsTeSet(data.Dataset):
    def __init__(self, root_dir, list_txt_path, config):
        self.input_size = config.TEST.IMAGE_SIZE # h, w
        self.root_dir = root_dir
        self.img_names_path = list_txt_path

        with open(self.img_names_path, 'r') as f:
            self.img_names = [x.strip() for x in f.readlines()]
        self.num_data = len(self.img_names)

    def __getitem__(self, item):
        img, img_size = self.load_img(os.path.join(self.root_dir, 'DUTS-TE-Image', self.img_names[item]))
        label_name = ''.join([os.path.splitext(self.img_names[item])[0], '.png'])
        label = load_label(os.path.join(self.root_dir, 'DUTS-TE-Mask', label_name))

        img = torch.Tensor(img)

        return {'img': img, 'name': self.img_names[item % self.num_data], 'size': img_size, 'label': label}

    def __len__(self):
        return self.num_data


    def load_img(self, path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = cv2.imread(path)
        in_ = np.array(im, dtype=np.float32)
        im_size = tuple(in_.shape[:2])
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = pad_uniform(in_, self.input_size[0], self.input_size[1])
        in_ = in_.transpose((2,0,1))
        return in_, im_size


def get_dataloader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        trainset = DutsTrSet(config.DATASET.TRAIN_ROOT, config.DATASET.TRAIN_LIST)
        train_loader = data.DataLoader(dataset=trainset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, 
        shuffle=shuffle, num_workers=config.WORKERS, pin_memory=pin)
        return train_loader
    else:
        dataset = DutsTeSet(config.DATASET.TEST_ROOT, config.DATASET.TEST_LIST, config)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.TEST.BATCH_SIZE_PER_GPU, 
        shuffle=shuffle, num_workers=config.WORKERS, pin_memory=pin)
        return data_loader


def load_train_img(path):
    # TODO: resize
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_


def load_img(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    # TODO: Load label image in  grayscale, not BGR format
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img, label


def get_uniform_pad_size_tblr(src, dst_h, dst_w):
    h_diff = dst_h - src.shape[0]
    w_diff = dst_w - src.shape[1]

    assert (h_diff > 0) and (w_diff > 0)
    
    pad_top = ceil(h_diff / 2)
    pad_bottom = h_diff - pad_top
    pad_left = ceil(w_diff / 2)
    pad_right = w_diff - pad_left

    return pad_top, pad_bottom, pad_left, pad_right


def pad_uniform(src, dst_h, dst_w):
    pad_top, pad_bottom, pad_left, pad_right = get_uniform_pad_size_tblr(src, dst_h, dst_w)
    
    return cv2.copyMakeBorder(src, pad_top, 
    pad_bottom, pad_left, 
    pad_right, cv2.BORDER_CONSTANT)