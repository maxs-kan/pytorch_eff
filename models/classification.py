from .base_model import BaseModel
from . import network
import torch
import torch.nn as nn
from utils import util

class ClassificationModel(BaseModel, nn.Module):
    
    def __init__(self, opt):
        super(ClassificationModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['cross_entropy']
        self.loss_names_test = ['cross_entropy'] 
                
        self.visuals_names = ['X']
        self.model_names = ['netG']
        self.netG = network.define_Gen(opt, self.device)
        
#         weight_class = torch.tensor([3.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(0.9, 0.999), weight_decay=opt.w_decay_G)
            self.optimizers.extend([self.optimizer_G])
            self.opt_names = ['optimizer_G']
    
    def set_input(self, input):
        self.X = input[0].to(self.device, non_blocking=self.opt.non_blocking and torch.cuda.is_available())
        self.y = input[1].to(self.device, non_blocking=self.opt.non_blocking and torch.cuda.is_available())
        
    def forward(self):
        #Depths
        self.logits = self.netG(self.X)
    
    def backward_G(self):
        self.loss_cross_entropy = self.criterion(self.logits, self.y)
        self.loss_cross_entropy.backward()

    def optimize_param(self):
        self.forward()
        self.zero_grad([self.netG])
        self.backward_G()
        self.optimizer_G.step()
        
    def calc_test_loss(self):
        with torch.no_grad():
            self.test_cross_entropy = self.criterion(self.logits, self.y)