"""
    setup model and datasets
"""


import copy

import numpy as np
import torch
from dataset import *
# from models import *
from VGG import *
from ResNet import *
from ResNets import *
from VGG_LTH import *



# from advertorch.utils import NormalizeByChannelMeanStd



model_dict = {
    "vgg16_bn_lth": vgg16_bn_lth,
    "vgg16_bn": vgg16_bn,
}



__all__ = ["setup_model_dataset"]


def setup_model_dataset(args):
    if args.dataset == "cifar10":    # for cipher10

        classes = 10    #this is for cipher10 [dataset]
        args.num_classes = 10    # this is added while trainng the cipher100 model 
        
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar100":

        classes = 100      #this is for cipher100 [dataset]
        args.num_classes = 100    

        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
    
    elif args.dataset == "svhn":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970]
        )

        train_set_loader, val_loader, test_loader = svhn_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers
        )

    
    elif args.dataset == "tinyimagenet":
        classes = 200
        args.num_classes = 200

        normalization = torch.nn.Identity()

        tiny_dataset = TinyImageNet(args)

        train_set_loader, val_loader, test_loader = tiny_dataset.data_loaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers
        )


    else:
        raise ValueError("Dataset not supprot yet !")

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    return model, train_set_loader, val_loader, test_loader


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)
