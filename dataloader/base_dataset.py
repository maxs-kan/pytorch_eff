import os
import glob
import imageio
import albumentations as A
import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    
    def __init__(self, opt):
        self.opt = opt
        self.IMG_EXTENSIONS = []
        self.transforms = []
    
    def add_extensions(self, ext_list):
        self.IMG_EXTENSIONS.extend(ext_list)    
    
    def is_image_files(self, files):
        for f in files:
            assert any(f.endswith(extension) for extension in self.IMG_EXTENSIONS), 'not implemented file extntion type {}'.format(f.split('.')[1])
        
    def get_paths(self, dir, reverse=False):
        files = []
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
        files = sorted(glob.glob(os.path.join(dir, '**/*.*'), recursive=True), reverse=reverse)
        return files[:min(self.opt.max_dataset_size, len(files))]
    
    def get_name(self, file_path):
        img_n = os.path.basename(file_path).split('.')[0]
        return img_n
    
    def read_data(self, path):
        return imageio.imread(path)
    
    @abstractmethod
    def __len__(self):
        return 0
    @abstractmethod
    def __getitem__(self, index):
        pass




