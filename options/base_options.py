import argparse
import os
from utils import util
import torch
import models
import dataloader


class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='test', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='If TRUE use CPU')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        #efficiency settings parameters for dataloading
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--pin_mem', action='store_true', default=False, help='Pin memory for dataloader')
        parser.add_argument('--non_blocking', action='store_true', default=False, help='How send to cuda')
#         parser.add_argument('--persistent_workers', action='store_true', default=False, help='Pin memory for dataloader')
#         parser.add_argument('--prefetch_factor', default=2, type=int, help='number of sample loaded in advance by each worker. 2 means there will be a total of 2 * num_workers samples prefetched across all workers.')
        
        parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic of cudnn, if true maybe slower')
        parser.add_argument('--use_dataparallel', action='store_true', default=False, help='DataParallel')
        
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=1, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--w_decay_G', type=float, default=0.0001, help='weight decat L2 reguralization for Gen')
        parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        
        # model parameters
        parser.add_argument('--model', type=str, default='classification', help='chooses which model to use. [classification]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--n_downsampling', type=int, default=1, help='# of downsamling')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB')
        parser.add_argument('--output_nc', type=int, default=10, help='# of classes')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the first conv layer for image')
        parser.add_argument('--n_blocks', type=int, default=6, help='# of res blocks')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none | group]')
        parser.add_argument('--dropout', action='store_true', default=False, help='dropout for the generator')
    
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='cifar10', help='chooses how datasets are loaded. [cifar10]')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.') 
        parser.add_argument('--no_data_shuffle', action='store_true', default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--H', type=int, default=32, help='Img size')
        parser.add_argument('--W', type=int, default=32, help='input batch size')

        # additional parameters
        parser.add_argument('--load_epoch', type=str, default='last', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--n_pic', type=int, default=3, help='# of picture pairs for vis.')
        parser.add_argument('--img_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--loss_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--debug', action='store_true', default=False, help='debug mode, no wandb')

        self.initialized = True 
        return parser

    def gather_options(self, isCodeCheck=False):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if isCodeCheck:
            opt, _ = parser.parse_known_args(args='')
        else:
            opt, _ = parser.parse_known_args()
        self.parser = parser
        return opt

    def print_options(self, opt):

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, isCodeCheck=False):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        
        opt = self.gather_options(isCodeCheck)
        opt.isTrain = opt.phase == 'train'
        if opt.isTrain:
            self.print_options(opt)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        self.opt = opt
        return self.opt
