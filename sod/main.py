import argparse
import os
from sod.dataset.dataset import get_loader
from sod.solver import Solver


def get_test_info(sal_mode='e'):
    if sal_mode == 'e':
        image_root = './data/ECSSD/Imgs/'
        image_source = './data/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = './data/PASCALS/Imgs/'
        image_source = './data/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = './data/DUTOMRON/Imgs/'
        image_source = './data/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = './data/HKU-IS/Imgs/'
        image_source = './data/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = './data/SOD/Imgs/'
        image_source = './data/SOD/test.lst'
    elif sal_mode == 't':
        image_root = './data/DUTS-TE/Imgs/'
        image_source = './data/DUTS-TE/test.lst'
    elif sal_mode == 'm_r': # for speed test
        image_root = './data/MSRA/Imgs_resized/'
        image_source = './data/MSRA/test_resized.lst'

    return image_root, image_source


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--cfg_path', type=str,
                        help='Path of .yaml file which contains model configuration')
    # Hyper-parameters
    # parser.add_argument('--n_color', type=int, default=3)
    # parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    # parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--cpu', dest='cuda', action='store_false')

    # Training settings
    # parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    # parser.add_argument('--pretrained_model', type=str)
    # parser.add_argument('--epoch', type=int, default=24)
    # parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    # parser.add_argument('--num_thread', type=int, default=1)
    # parser.add_argument('--load', type=str, default='')
    parser.add_argument('--weights_save_dir', type=str, default='./results', 
                        help='Directory to save weights')
    parser.add_argument('--weights_save_cycle', type=int, default=3,   
                        help='Save weights every $epoch_save epochs during training')
    parser.add_argument('--acc_step_size', type=int, default=10,
                        help='Accumulate gradients through $acc_step_size iterations')
    # parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='', 
                        help='Root directory of training set')
    parser.add_argument('--train_list', type=str, default='', 
                        help='Text file which contains (image filename, ground-truth filename) pairs of training set')

    # Testing settings
    parser.add_argument('--model', type=str, default=None) # Snapshot
    parser.add_argument('--test_fold', type=str, default=None) # Test results saving folder
    # parser.add_argument('--sal_mode', type=str, default='e') # Test image dataset

    args = parser.parse_args()

    if not os.path.exists(args.save_folder): os.makedirs(args.save_folder)

    # # Get test set info
    # test_root, test_list = get_test_info(args.sal_mode)
    # args.test_root = test_root
    # args.test_list = test_list

    return args


def main(args):
    if args.mode == 'train':
        train_loader = get_loader(args)
        run = 0
        while os.path.exists("%s/run-%d" % (args.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (args.save_folder, run))
        os.mkdir("%s/run-%d/models" % (args.save_folder, run))
        args.save_folder = "%s/run-%d" % (args.save_folder, run)
        solver = Solver(train_loader, None, args)
        solver.train()
    elif args.mode == 'test':
        args.test_root, args.test_list = get_test_info(args.sal_mode)
        test_loader = get_loader(args, mode='test')
        if not os.path.exists(args.test_fold): os.mkdir(args.test_fold)
        solver = Solver(None, test_loader, args)
        solver.test()


if __name__ == '__main__':
    args = parse_args()
    main(args)
