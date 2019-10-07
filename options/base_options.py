import argparse
import os
import torch
from util import util
from datetime import datetime

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--root_dir', nargs='+', help='root directions of train data')
        self.parser.add_argument('--component', type=str, default='Decoder')
        self.parser.add_argument('--batchSize', type=int, default=1)
        self.parser.add_argument('--nThreads', type=int, default=2)
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--drop_last', action='store_true', help='drop last')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')

        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--print_freq', type=int, default=100)
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--rotate_range', type=int, default=20, help='rotate range of data augmentation')
        self.parser.add_argument('--translate_range', type=int, default=20, help='translation range of data augmentation')
        self.parser.add_argument('--zoom_range', type=float, nargs='+', help='zoom range of data augmentation')
        self.parser.add_argument('--mirror', dest='mirror', action='store_true')
        self.parser.add_argument('--normalise', dest='normalise', action='store_true')
        self.parser.add_argument('--normalisation_type', type=str, help='normalisation type')

        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')

        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal|transformer]')

        self.parser.add_argument('--pretrain_root', type=str, default='./pretrained_models')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')

        self.initialized = True

    def get_time(self,):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # make dirs

        self.opt.model_name = "{}_{}".format(self.opt.component, self.get_time())
        self.opt.name = self.opt.model_name
        if self.isTrain:
            self.opt.model_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            for path in [self.opt.checkpoints_dir, self.opt.model_dir]:
                util.mkdirs(path)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.isTrain:
            file_name = os.path.join(self.opt.model_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
