# util/data_loader.py

import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}


class RK4Dataset(Dataset):
    """Wrap your .npy containing {'prediction', 'Truth', ...} for OOD thresholding."""
    def __init__(self, npy_file, skip=1):
        data = np.load(npy_file, allow_pickle=True).item()
        self.pred  = data['prediction'][::skip]   # shape (T, input_dim)
        self.truth = data['Truth'][::skip]        # shape (T, input_dim)

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, i):
        # return (input, label) as float tensors
        return (torch.from_numpy(self.pred[i]).float(),
                torch.from_numpy(self.truth[i]).float())


def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
    })[config_type]

    train_loader = val_loader = None
    lr_schedule, num_classes = [50, 75, 90], 0

    # --- image datasets ---
    if args.in_dataset == "CIFAR-10":
        num_classes = 10
        if 'train' in split:
            ds = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True,
                transform=config.transform_train)
            train_loader = DataLoader(ds, batch_size=config.batch_size,
                                      shuffle=True, **kwargs)
        if 'val' in split:
            ds = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=transform_test)
            val_loader = DataLoader(ds, batch_size=config.batch_size,
                                    shuffle=False, **kwargs)

    elif args.in_dataset == "CIFAR-100":
        num_classes = 100
        if 'train' in split:
            ds = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True,
                transform=config.transform_train)
            train_loader = DataLoader(ds, batch_size=config.batch_size,
                                      shuffle=True, **kwargs)
        if 'val' in split:
            ds = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=transform_test)
            val_loader = DataLoader(ds, batch_size=config.batch_size,
                                    shuffle=False, **kwargs)

    elif args.in_dataset.lower() == "imagenet":
        root = 'datasets/id_data/imagenet'
        num_classes = 1000
        if 'train' in split:
            ds = torchvision.datasets.ImageFolder(
                os.path.join(root, 'train'),
                transform=config.transform_train_largescale)
            train_loader = DataLoader(ds, batch_size=config.batch_size,
                                      shuffle=True, **kwargs)
        if 'val' in split:
            ds = torchvision.datasets.ImageFolder(
                os.path.join(root, 'val'),
                transform=config.transform_test_largescale)
            val_loader = DataLoader(ds, batch_size=config.batch_size,
                                    shuffle=False, **kwargs)

    # --- time‐series RK4 dataset for thresholding ---
    elif args.in_dataset.lower() == 'rk4':
        # args.input_file should point to your .npy from the RK4 script
        ds = RK4Dataset(args.input_file, skip=getattr(args, 'skip', 1))
        # no train/val split—just use 'val' split for threshold estimation
        if 'val' in split:
            val_loader = DataLoader(ds,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    **kwargs)
        num_classes = None

    else:
        raise ValueError(f"Unsupported in_dataset: {args.in_dataset}")

    return EasyDict({
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "lr_schedule":  lr_schedule,
        "num_classes":  num_classes,
    })


def get_loader_out(args, dataset=(''), config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_train_largescale': transform_train_largescale,
            'transform_test_largescale':  transform_test_largescale,
            'batch_size':       args.batch_size
        },
    })[config_type]

    train_ood_loader = val_ood_loader = None

    # ... (unchanged existing OOD branches) ...

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader":   val_ood_loader,
    })
