import os 
import torch
from abc import ABC, abstractmethod
from collections import OrderedDict
from models import network
class BaseModel(ABC, torch.nn.Module):    
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        if not opt.use_cpu:
            assert(torch.cuda.is_available())
        self.device = torch.device('cuda:{}'.format(torch.cuda.current_device()) if  not opt.use_cpu else 'cpu')
        print('Runing on {}'.format(self.device))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        self.loss_names = []
        self.model_names = []
        self.visuals_names = []
        self.optimizers = []
#         self.image_paths = []
        self.metric = 0    # validation loss plateau sheduler
    
    @abstractmethod
    def set_input(self, input): 
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def optimize_param(self):
        pass
    
    def setup(self):
        if self.isTrain:
            self.schedulers = [network.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        if not self.isTrain or self.opt.continue_train:
            load_suffix = 'iter_%d' %  self.opt.load_epoch
            self.load_networks(load_suffix)
        self.print_networks()
    
    def train_mode(self):
        for name in self.model_names:
            assert isinstance(name, str), 'model_names  must be string'
            net = getattr(self, name)
            net.train()
    
    def eval(self):
        for name in self.model_names:
            assert isinstance(name, str), 'model_names  must be string'
            net = getattr(self, name)
            net.eval()
   
    def test(self):
        self.eval()
        with torch.no_grad():
            self.forward()
    
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
    
    def get_current_vis(self):
        visual_dict = OrderedDict()
        for name in self.visuals_names:
            assert isinstance(name, str), 'visual name must be str'
            visual_dict[name] = getattr(self, name)
        return visual_dict
    
    def get_current_losses(self):
        loss_dict = OrderedDict()
        for name in self.loss_names:
            assert isinstance(name, str), 'loss name must be str'
            loss_dict[name] = float(getattr(self, 'loss_' + name))
        return loss_dict
    
    def get_current_losses_test(self):
        loss_dict = OrderedDict()
        for name in self.loss_names_test:
            assert isinstance(name, str), 'loss name must be str'
            loss_dict['test_' + name] = float(getattr(self, 'test_' + name))
        return loss_dict
    
    def save_net(self, epoch):
        state_dict = {}
        save_fname = '%s.pt' % (epoch)
        save_path = os.path.join(self.save_dir, save_fname)
        for name in self.model_names:
            assert isinstance(name, str), 'name must be str'
            net = getattr(self, name) 
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                state_dict[name] = net.module.cpu().state_dict()
                net.cuda()                             
            elif torch.cuda.is_available():
                state_dict[name] = net.cpu().state_dict()
                net.cuda()
            else:
                state_dict[name] = net.state_dict()
        for name in self.opt_names:
            assert isinstance(name, str), 'name must be str'
            opt = getattr(self, name)
            state_dict[name] = opt.state_dict()
        torch.save(state_dict, save_path)
    
    def load_networks(self, epoch):
        load_filename = '%s.pt' % (epoch)
        load_path = os.path.join(self.save_dir, load_filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        for name in self.model_names:
            assert isinstance(name, str), 'model name must be str'
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = checkpoint[name]
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
        if self.isTrain:
            for name in self.opt_names:
                assert isinstance(name, str), 'model name must be str'
                opt = getattr(self, name)
                print('loading the optimizer from %s' % load_path)
                opt.load_state_dict(checkpoint[name])
    
    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            assert isinstance(name, str), 'model name mus be str'
            net = getattr(self, name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    
    def zero_grad(self, nets):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.grad = None
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
        
#     def debug_grad(self):
#         for n in self.model_names:
#             net = getattr(self, n)
#             for name, param in net.named_parameters():
#                 if not torch.isfinite(param.grad).all():
#                         print('NaN in {} parameters gradient of {}'.format(name, n))
#                         self.save_log()
    
#     def debug_weighs(self):
#         for n in self.model_names:
#             net = getattr(self, n)
#             for name, param in net.named_parameters():
#                 if not torch.isfinite(param).all():
#                         print('NaN in {} parameters of {}'.format(name, n))
#                         self.save_log()