from options.base_options import BaseOptions 
from dataloader import create_dataset
from models import create_model
# from utils.visualizer import Visualizer
# import matplotlib.pyplot as plt
# import wandb
import time
import numpy as np
import random
import torch
import os
import copy
from collections import OrderedDict 

def acc_loss(d_acc, d):
    output = OrderedDict([(key, d_acc[key] + d[key]) for key in d_acc.keys()])
    return output
def div_loss(d_acc, n, epoch):
    output = OrderedDict([(key, d_acc[key] / n) for key in d_acc.keys()])
    return output

if __name__ == '__main__':
    seed_value = 101
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    epoch_time_list = []
    epoch_val_time_list = []
    
    opt = BaseOptions().parse()   # get training options
    opt_v = copy.deepcopy(opt)
    opt_v.isTrain = False
    opt_v.phase = 'val'
    torch.cuda.set_device(opt.gpu_ids[0])
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.backends.cudnn.benchmark = not opt.deterministic
#     torch.autograd.set_detect_anomaly(True)
    
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = {}'.format(dataset_size))
    print()
    dataset_v = create_dataset(opt_v)  
    dataset_size_v = len(dataset_v)    # get the number of images in the dataset.
    print('The number of val images = {}'.format(dataset_size_v))
    print()
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    global_iter = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        print()
        print('Training...')
        model.train_mode()
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            global_iter += 1
            model.set_input(data)
            model.optimize_param()
#             torch.cuda.empty_cache()
            iter_finish_time = time.time()
#             if global_iter % opt.loss_freq == 0:
                
#             if global_iter % opt.img_freq == 0:
        epoch_time = time.time() - epoch_start_time 
        epoch_time_list += [epoch_time]
        print('End of training on epoch {} / {} \t Time Taken: {:04.2f} sec'.format(epoch, opt.n_epochs + opt.n_epochs_decay, epoch_time))
        print('Validation...')
        n_b = 0
        epoch_val_start_time = time.time()
        for i, data in enumerate(dataset_v):
            n_b += 1
            model.set_input(data)
            model.test()
            model.calc_test_loss()
            if i == 0:
                mean_loss = model.get_current_losses_test()
            else:
                mean_loss = acc_loss(mean_loss, model.get_current_losses_test())
        epoch_val_time = time.time() - epoch_val_start_time
        epoch_val_time_list += [epoch_val_time]
        print('End of validation on epoch {}  \t Time Taken: {:04.2f} sec'.format(epoch, epoch_val_time))  
        mean_loss = div_loss(mean_loss, n_b, epoch)
        print('Validation loss {}'.format(mean_loss['test_cross_entropy']))

#         if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
#             print('saving the model at the end of epoch {}, iters {}'.format(epoch, global_iter))
#             model.save_net(epoch)
        model.update_learning_rate()
#     model.save_net('last')
    epoch_time_list = np.array(epoch_time_list)
    epoch_val_time_list = np.array(epoch_val_time_list)
    print('Finish')
    
    