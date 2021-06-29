import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from sod.networks.poolnet import build_model, weights_init
from sod.dataset.dataset import load_image_test
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time

import lib.models
from lib.config import config_sod_ddrnet_23_slim


class Solver(object):
    def __init__(self, train_loader, test_loader, args):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config_sod_ddrnet_23_slim()
        self.config.merge_from_file(args.cfg_path)
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.acc_step_size = args.acc_step_size
        self.show_every = args.show_every
        self.lr_decay_epoch = [15,]
        self.build_model()
        self.print_network(self, self.net, 'DDRNet')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        if self.args.cuda:
            cudnn.benchmark = self.config.CUDNN.BENCHMARK
            cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
            cudnn.enabled = self.config.CUDNN.ENABLED

        # build model
        if torch.__version__.startswith('1'):
            module = lib.models.__dict__[self.config.MODEL.NAME]
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.SyncBatchNorm    
        self.net = lib.models.__dict__[self.config.MODEL.NAME].__dict__['get_seg_model'](self.config).to(self.device)

        # load pre-trained weights
        pretrained_dict = torch.load(self.args.TEST.MODEL_FILE, map_location=self.device)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.net.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        if self.args.mode == 'train': 
            self.net.train()
            self.lr = self.config.TRAIN.LR
            self.wd = self.config.TRAIN.WD
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        elif self.args.mode == 'test': 
            self.net.eval()


    def infer(self, input_path, output_path):
        input_data, _ = load_image_test(input_path)
        preds = self.net(input_data)
        pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
        multi_fuse = 255 * pred
        cv2.imwrite(output_path, multi_fuse)


    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.args.cuda:
                    images = images.cuda()
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(self.args.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')


    def train(self):
        iter_num = len(self.train_loader.dataset) // self.args.batch_size
        aveGrad = 0
        for epoch in range(self.args.epoch):
            r_sal_loss= 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label= Variable(sal_image), Variable(sal_label)
                if self.args.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                ''' TODO: Analyze the line below
                It's fair enough to divide the losses by size of the mini-batch
                But what is $self.iter_size?
                See the `self.optimizer.step()`
                It optimizes the model once every $self.iter_size iterations rather than each iteration
                So, it divides the losses by not only batch size but also $self.iter_size
                However, it seems the losses are averaged across observations for each minibatch if `reduction='mean'`
                '''
                sal_loss = sal_loss_fuse / (self.acc_step_size * self.args.batch_size)
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                '''
                What is DSS?
                dynamic stochastic sequence?
                Find gradient accumulation for more details
                https://makinarocks.github.io/Gradient-Accumulation/
                '''
                if aveGrad % self.acc_step_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.args.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.args.epoch, i, iter_num, r_sal_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0

            if (epoch + 1) % self.args.weights_save_cycle == 0:
                pth_path = '{}/{}/{}_epoch_{}.pth'.format(
                    self.args.weights_save_dir, self.configs.DATASET.NAME,
                    self.configs.MODEL.NAME, epoch + 1)
                pth_dir = os.path.dirname(pth_path)
                if not os.path.exists(pth_dir): os.path.makedirs(pth_dir)
                torch.save(self.net.state_dict(), pth_path)

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.args.save_folder)
