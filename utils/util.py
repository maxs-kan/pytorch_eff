import torch
import numpy as np
import os

def logits_to_label(input):
    prob = F.softmax(input,dim=1)
    label = torch.argmax(prob, dim=1)
    return label.data.cpu().numpy()

def torch2np(input):
    return input.cpu().permute(0,2,3,1).numpy()[:,:,:,0]

def tensor2im(input, opt, isDepth = True):

    if not isinstance(input, np.ndarray):
        if isinstance(input, torch.Tensor):  # get the data from a variable
            tensor = input.data
        else:
            return input
        tensor = tensor * 127.5 + 127.5
        numpy = tensor.cpu().permute(0,2,3,1).numpy().astype(np.uint8)
    else:  # if it is a numpy array, do nothing
        numpy = input
    return numpy

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
