import os
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import random
from lib.utils.utils import letterbox_resize


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


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_img(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_dataloader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = DutsTrSet(config.DATASET.TRAIN_ROOT, config.DATASET.TRAIN_LIST)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, 
        shuffle=shuffle, num_workers=config.WORKERS, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.DATASET.TEST_ROOT, config.DATASET.TEST_LIST)
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
    in_ = in_.transpose((2,0,1))
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
