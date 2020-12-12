import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
        
def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'group':
        norm_layer = lambda n_ch : nn.GroupNorm(num_groups=8, num_channels=n_ch, affine=True)#functools.partial(nn.GroupNorm, num_groups=8, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain='relu', param=None):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init.calculate_gain(init_gain, param))#
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init.calculate_gain(init_gain, param))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None: 
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (m.weight is not None) and (classname.find('Norm') != -1): 
            init.normal_(m.weight.data, mean=1.0, std=0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain='relu', gpu_ids=[], param=None, data_parallel=True, device='cpu'):

    net.to(device)
    if len(gpu_ids) > 0:
        if data_parallel:
            net = torch.nn.DataParallel(net, gpu_ids).cuda()
            print('Send network on {} gpu'.format(gpu_ids))
    init_weights(net=net, init_type=init_type, init_gain=init_gain, param=param)
    return net

def define_Gen(opt, device):
    use_bias = opt.norm == 'instance'
    net = Generator(opt, use_bias)
    return init_net(net=net, init_type=opt.init_type, init_gain='relu', gpu_ids=opt.gpu_ids, data_parallel=opt.use_dataparallel, device=device)

class Generator(nn.Module):
    def __init__(self, opt, use_bias):
        super(Generator, self).__init__()
        self.opt = opt
        norm_layer = get_norm_layer(norm_type=opt.norm)
        self.enc = Encoder(input_nc=opt.input_nc, base_nc=opt.ngf, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
        self.bottlenec = ResnetBottlenec(base_nc=opt.ngf, n_blocks=opt.n_blocks, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
        self.dec = Decoder(base_nc=opt.ngf, output_nc=opt.output_nc, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
    
    def forward(self, x):
        x = self.enc(x)
        x = self.bottlenec(x)
        logits = self.dec(x)
        return logits   

class Encoder(nn.Module):
    def __init__(self, input_nc, base_nc, norm_layer, use_bias,  opt):
        super(Encoder, self).__init__()
        model = [nn.Conv2d(input_nc, base_nc, kernel_size=7, stride=1, padding=3, dilation=1, padding_mode='replicate', bias=use_bias),
                 norm_layer(base_nc),
                 nn.ReLU(True)]
        for i in range(opt.n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(base_nc * mult, base_nc * mult * 2, kernel_size=4, stride=2, padding=1, dilation=1, padding_mode='replicate', bias=use_bias),
                      norm_layer(base_nc * mult * 2),
                      nn.ReLU(True)]
            self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self, base_nc, output_nc, norm_layer, use_bias, opt):
        super(Decoder, self).__init__()
        model = []
        mult = 2**opt.n_downsampling
        model +=[nn.Flatten(),
                 nn.Linear(base_nc * mult * (opt.H//mult) * (opt.W//mult), 100),
                 nn.ReLU(True),
                ]
        if opt.dropout:
            model += [nn.Dropout(0.5)]
        model +=[nn.Linear(100, 10)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

    
class ResnetBottlenec(nn.Module):
    def __init__(self, base_nc, n_blocks, norm_layer, use_bias, opt, use_dilation=False):
        super(ResnetBottlenec, self).__init__()       
        model = []
        mult = 2**opt.n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            if use_dilation:
                dilation = min(2**i, 8)
            else:
                dilation = 1
            model += [ResnetBlock(dim=base_nc * mult, dilation=dilation, norm_layer=norm_layer, use_bias=use_bias, opt=opt)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer, use_bias, opt):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dilation, norm_layer, use_bias, opt)

    def build_conv_block(self, dim, dilation, norm_layer, use_bias, opt):
        conv_block = []
        pad = int(dilation * ( 3 - 1) / 2) 
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=pad, dilation=dilation, padding_mode='replicate', bias=use_bias),
                       norm_layer(dim), 
                       nn.ReLU(True)]
        if opt.dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=pad, dilation=dilation, padding_mode='replicate', bias=use_bias), 
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out       