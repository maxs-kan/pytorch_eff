import torch.utils.data
from dataloader.base_dataset import BaseDataset
import torchvision
import torchvision.transforms as transforms

def create_dataset(opt):
    dataset = Dataset_Dataloader(opt)
    return dataset


class Dataset_Dataloader():
    def __init__(self, opt):
        self.opt = opt
        if opt.dataset_mode == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                        train=opt.isTrain,
                                                        download=True, 
                                                        transform=transform)
        else:
            raise NotImplementedError('Implement dataset')
        
        print('Dataset {} was created'.format(type(self.dataset).__name__))
        if (min(len(self.dataset), self.opt.max_dataset_size) % opt.batch_size != 0) and opt.isTrain:
            print('Warning, drop last batch')
            drop_last = True
        else:
            drop_last = False
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle= not opt.no_data_shuffle,
            num_workers=int(opt.num_workers),
            drop_last=drop_last,
            pin_memory=opt.pin_mem,
        )
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

