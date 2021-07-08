import torch
from torch.jit import Error
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
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sod.dataset.dataset import get_uniform_pad_size_tblr


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
        # self.feature_extraction_keys = ['seghead_extra.conv2.weight', 'seghead_extra.conv2.bias',
        #                                 'final_layer.conv2.weight', 'final_layer.conv2.bias']
        # self.feature_extraction_blocks = ['final_layer']
        self.is_feature_extraction = False
        self.maes_through_epochs = list()
        self.build_model()
        # self.print_network(self.net, 'DDRNet')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))


    def set_parameter_requires_grad(self, is_train):
        # if not self.is_feature_extraction:
        #     for param in self.net.parameters():
        #         param.requires_grad = True
        #     return 0
        # for name, param in self.net.named_parameters():
        #     block = name.split('.')[0]
        #     if block in self.feature_extraction_blocks:
        #         param.requires_grad = True
        #     elif not block in self.feature_extraction_blocks:
        #         param.requires_grad = False
        #     else: raise Error
        for param in self.net.parameters():
            param.requires_grad = is_train
        return 0


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
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.SyncBatchNorm    
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

        self.net = torch.nn.DataParallel(self.net, device_ids=self.config.GPUS).cuda()

        if self.args.mode == 'train': 
            self.net.train()
            self.set_parameter_requires_grad(is_train=True)
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


    def evaluate(self):
        self.net.eval()
        self.set_parameter_requires_grad(is_train=False)
        if not os.path.exists(self.config.OUTPUT_DIR): os.makedirs(self.config.OUTPUT_DIR)
        mae = 0.0
        for i, minibatch in enumerate(tqdm(self.test_loader)):
            img, label = minibatch['img'], np.squeeze(minibatch['label'].cpu().data.numpy())
            name = minibatch['name']

            with torch.no_grad():
                img = Variable(img).to(self.device)
                pred = self.net(img)[0]

                h_pred, w_pred = pred.size(2), pred.size(3)
                h_img, w_img = img.size(2), img.size(3)
                if (h_pred != h_img) or (w_pred != w_img):
                    pred = F.interpolate(input=pred, size=(h_img, w_img), mode='bilinear', 
                                         align_corners=self.config.MODEL.ALIGN_CORNERS)
                                         
                pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy())

                pad_top, pad_bottom, pad_left, pad_right = get_uniform_pad_size_tblr(label, 
                self.config.TEST.IMAGE_SIZE[0], 
                self.config.TEST.IMAGE_SIZE[1])
                pred = pred[pad_top:pred.shape[0]-pad_bottom, 
                pad_left:pred.shape[1]-pad_right]
                
                mae += mean_absolute_error(label, pred)
                # res = 255 * pred
                # cv2.imwrite(os.path.join(self.config.OUTPUT_DIR, ''.join([name[0][:-4], '.png'])), res)
        mae /= len(self.test_loader.dataset)
        return mae


    def get_mae(self, pred, label):
        pred = torch.sigmoid(pred).cpu().data.numpy()
        label = label.cpu().data.numpy()
        mae = np.average(np.abs(pred - label))
        return mae


    def train(self):
        eval_res_path = '{}/{}/{}_MAE.txt'.format(
            self.config.TRAIN.WEIGHTS_SAVE_DIR, self.config.DATASET.NAME,
            self.config.MODEL.NAME)
        eval_res_dir = os.path.dirname(eval_res_path)
        if not os.path.exists(eval_res_dir): os.makedirs(eval_res_dir)
        eval_res_file = open(eval_res_path, 'w')

        current_grad_acc_step = 0

        for epoch in range(self.config.TRAIN.BEGIN_EPOCH, self.config.TRAIN.END_EPOCH):
            loss_per_grad_acc= 0
            train_mae = 0.0
            self.net.zero_grad()
            print('-------------------------------')
            for i, minibatch in enumerate(tqdm(self.train_loader)):
                img, label = minibatch['img'], minibatch['label']
                if img is None: continue
                if (img.size(2) != label.size(2)) or (img.size(3) != label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                img, label= Variable(img), Variable(label)
                img, label = img.to(self.device), label.to(self.device)

                loss_minibatch = torch.zeros([1]).to(self.device)
                preds = self.net(img)
                for aux_loss_idx, (balance_weight, pred) in enumerate(zip(self.config.LOSS.BALANCE_WEIGHTS, preds)):
                    h_pred, w_pred = pred.size(2), pred.size(3)
                    h_label, w_label = label.size(2), label.size(3)
                    if (h_pred != h_label) or (w_pred != w_label):
                        pred = F.interpolate(input=pred, size=(h_label, w_label), mode='bilinear', 
                                            align_corners=self.config.MODEL.ALIGN_CORNERS)

                    loss_minibatch += balance_weight * F.binary_cross_entropy_with_logits(
                        pred, label, reduction='mean')

                    if aux_loss_idx != 0: continue
                    train_mae += self.get_mae(pred, label) * img.size(0)

                ''' TODO: Analyze the line below
                It's fair enough to divide the losses by size of the mini-batch
                But what is $self.iter_size?
                See the `self.optimizer.step()`
                It optimizes the model once every $self.iter_size iterations rather than each iteration
                So, it divides the losses by not only batch size but also $self.iter_size
                However, it seems the losses are averaged across observations for each minibatch if `reduction='mean'`
                '''
                loss_minibatch_unit_grad_acc = loss_minibatch / (self.grad_accumulation_step_size)
                loss_per_grad_acc += loss_minibatch_unit_grad_acc.data

                loss_minibatch_unit_grad_acc.backward()

                current_grad_acc_step += 1

                # accumulate gradients as done in DSS
                '''
                What is DSS?
                dynamic stochastic sequence?
                Find gradient accumulation for more details
                https://makinarocks.github.io/Gradient-Accumulation/
                '''
                if current_grad_acc_step % self.grad_accumulation_step_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    current_grad_acc_step = 0

                # if i % (self.print_freq // self.config.TRAIN.BATCH_SIZE_PER_GPU) == 0:
                #     if i == 0:
                #         x_showEvery = 1
                #     print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                #         epoch, self.config.TRAIN.END_EPOCH, i, iter_num, r_sal_loss/x_showEvery))
                #     print('Learning rate: ' + str(self.lr))
                #     r_sal_loss= 0


            train_mae /= len(self.train_loader.dataset)
            test_mae = self.evaluate()
            print('epoch: [%2d/%2d]' % (
                epoch, self.config.TRAIN.END_EPOCH))
            print('Learning rate: ' + str(self.lr))
            loss_per_grad_acc= 0
            print('{:>16} {:>16} {:>16}'.format('', 'DUTS-TR', 'DUTS-TE'))
            print('{:>16} {:>16.3f} {:>16.3f}'.format('MAE', train_mae, test_mae))
            print('\n')
            eval_res_file.write('{} {}\n'.format(train_mae, test_mae))
            self.net.train()
            self.set_parameter_requires_grad(is_train=True)

            if (epoch + 1) % self.config.TRAIN.WEIGHTS_SAVE_FREQ == 0:
                pth_path = '{}/{}/{}_epoch_{}.pth'.format(
                    self.config.TRAIN.WEIGHTS_SAVE_DIR, self.config.DATASET.NAME,
                    self.config.MODEL.NAME, epoch + 1)
                pth_dir = os.path.dirname(pth_path)
                if not os.path.exists(pth_dir): os.makedirs(pth_dir)
                torch.save(self.net.state_dict(), pth_path)

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '{}/{}/{}_final.pth'.format(
            self.config.TRAIN.WEIGHTS_SAVE_DIR, self.config.DATASET.NAME,
            self.config.MODEL.NAME))

        eval_res_file.close()