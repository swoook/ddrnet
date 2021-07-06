import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from sod.dataset.dataset import load_img
import numpy as np
import os
import cv2
import time
from collections import OrderedDict
import lib.models


class Solver(object):
    def __init__(self, train_loader, test_loader, config, args):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.config = config
        self.device = torch.device(
            'cuda:{}'.format(self.args.local_rank) if ((len(self.config.GPUS) > 0) & (torch.cuda.is_available())) 
            else 'cpu')
        self.grad_accumulation_step_size = self.config.TRAIN.GRAD_ACCUMULATION_STEP_SIZE
        self.print_freq = self.config.TRAIN.PRINT_FREQ
        self.lr_decay_epoch = [15,]
        self.feature_extraction_keys = ['seghead_extra.conv2.weight', 'seghead_extra.conv2.bias',
                                        'final_layer.conv2.weight', 'final_layer.conv2.bias']
        self.build_model()
        self.print_network(self.net, 'DDRNet')

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
        ''' TODO: Is this block necessary?
        https://pytorch.org/docs/stable/backends.html#torch-backends-cudnn
        '''
        if self.device.type == 'cuda':
            cudnn.benchmark = self.config.CUDNN.BENCHMARK
            cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
            cudnn.enabled = self.config.CUDNN.ENABLED
            ''' TODO: Is `torch.cuda.set_device` necessary?
            https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html#torch.cuda.set_device
            > Usage of this function is discouraged in favor of device. 
            > In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable.
            '''
            # torch.cuda.set_device(self.device)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://",)  

        # build model
        if torch.__version__.startswith('1'):
            module = lib.models.__dict__[self.config.MODEL.NAME]
            ''' TODO
            torch.nn.SyncBatchNorm requires calling `torch.distributed.init_process_group` while training
            '''
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d    
        self.net = module.__dict__['get_seg_model'](self.config, 
        pretrained=False, num_classes=1).to(self.device)

        # load pre-trained weights
        pretrained_dict = torch.load(self.config.MODEL.PRETRAINED, map_location=self.device)
        if 'state_dict' in pretrained_dict: pretrained_dict = pretrained_dict['state_dict']
        # https://stackoverflow.com/a/50872567
        # https://www.daleseo.com/python-collections-ordered-dict/#%EB%8F%99%EB%93%B1%EC%84%B1-%EB%B9%84%EA%B5%90
        pretrained_dict = OrderedDict({k[6:]: v for k, v in pretrained_dict.items() if (k[:6] == 'model.')})
        model_dict = self.net.state_dict()
        # https://github.com/pytorch/pytorch/issues/40859#issuecomment-857936621
        pretrained_dict = OrderedDict({k: v if model_dict[k].size() == v.size() else model_dict[k]
                                       for k, v in zip(model_dict.keys(), pretrained_dict.values())})
        
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
        input_data, _ = load_img(input_path)
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
                images = Variable(images).to(self.device)
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(self.args.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')


    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.TRAIN.BATCH_SIZE_PER_GPU
        aveGrad = 0
        for epoch in range(self.config.TRAIN.BEGIN_EPOCH, self.config.TRAIN.END_EPOCH):
            r_sal_loss= 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['img'], data_batch['label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label= Variable(sal_image), Variable(sal_label)
                sal_image, sal_label = sal_image.to(self.device), sal_label.to(self.device)

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
                sal_loss = sal_loss_fuse / (self.grad_accumulation_step_size * self.args.batch_size)
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
                if aveGrad % self.grad_accumulation_step_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.print_freq // self.args.batch_size) == 0:
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
