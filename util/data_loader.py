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
        self.pred  = data['prediction'][::skip]   # shape (T, L)
        self.truth = data['Truth'][::skip]        # shape (T, L)

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, i):
        return (torch.from_numpy(self.pred[i]).float(),
                torch.from_numpy(self.truth[i]).float())


def get_loader_in(args, config_type='default', split=('train', 'val')):
    # allow split='val' shorthand
    if isinstance(split, str):
        split = (split,)

    config = EasyDict({
        "default": {
            'transform_train':        transform_train,
            'transform_test':         transform_test,
            'transform_train_largescale': transform_train_largescale,
            'transform_test_largescale':  transform_test_largescale,
            'batch_size':             args.batch_size,
        },
    })[config_type]

    train_loader = None
    val_loader   = None
    num_classes  = 0
    lr_schedule  = [50, 75, 90]

    # CIFAR-10
    if args.in_dataset == "CIFAR-10":
        num_classes = 10
        if 'train' in split:
            ds = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                              transform=config.transform_train)
            train_loader = DataLoader(ds, batch_size=args.batch_size,
                                      shuffle=True, **kwargs)
        if 'val' in split:
            ds = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                              transform=transform_test)
            val_loader = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=False, **kwargs)

    # CIFAR-100
    elif args.in_dataset == "CIFAR-100":
        num_classes = 100
        if 'train' in split:
            ds = torchvision.datasets.CIFAR100('./data', train=True, download=True,
                                               transform=config.transform_train)
            train_loader = DataLoader(ds, batch_size=args.batch_size,
                                      shuffle=True, **kwargs)
        if 'val' in split:
            ds = torchvision.datasets.CIFAR100('./data', train=False, download=True,
                                               transform=transform_test)
            val_loader = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=False, **kwargs)

    # ImageNet
    elif args.in_dataset.lower() == "imagenet":
        root = 'datasets/id_data/imagenet'
        num_classes = 1000
        if 'train' in split:
            ds = torchvision.datasets.ImageFolder(os.path.join(root, 'train'),
                                                  transform=config.transform_train_largescale)
            train_loader = DataLoader(ds, batch_size=args.batch_size,
                                      shuffle=True, **kwargs)
        if 'val' in split:
            ds = torchvision.datasets.ImageFolder(os.path.join(root, 'val'),
                                                  transform=config.transform_test_largescale)
            val_loader = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=False, **kwargs)

    # RK4 time‑series
    elif args.in_dataset.lower() == 'rk4':
        num_classes = None
        ds = RK4Dataset(args.input_file, skip=getattr(args, 'skip', 1))
        if 'val' in split:
            val_loader = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=False, **kwargs)

    else:
        raise ValueError(f"Unsupported in_dataset: {args.in_dataset}")

    return EasyDict({
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "lr_schedule":  lr_schedule,
        "num_classes":  num_classes,
    })


def get_loader_out(args, dataset=('',''), config_type='default', split=('train', 'val')):
    # allow split='val' shorthand
    if isinstance(split, str):
        split = (split,)

    config = EasyDict({
        "default": {
            'transform_train':        transform_train,
            'transform_test':         transform_test,
            'transform_train_largescale': transform_train_largescale,
            'transform_test_largescale':  transform_test_largescale,
            'batch_size':             args.batch_size,
        },
    })[config_type]

    train_ood_loader = None
    val_ood_loader   = None

    out_name = dataset[1].lower()

    # ---- existing image‐OOD branches here ----
    # e.g. SVHN, DTD, CIFAR-100, etc., exactly as React originally did
    # (you can leave those unchanged)

    # ---- noise_series time‑series OOD ----
    if 'val' in split and out_name == 'noise_series':
        data = np.load(args.input_file, allow_pickle=True).item()
        T, L = data['prediction'].shape

        class NoiseSeriesDataset(Dataset):
            def __init__(self, T, L):
                self.T = T
                self.L = L
            def __len__(self):
                return self.T
            def __getitem__(self, idx):
                x = np.random.randn(self.L).astype(np.float32)
                y = np.zeros(self.L, dtype=np.float32)
                return (torch.from_numpy(x), torch.from_numpy(y))

        ds = NoiseSeriesDataset(T, L)
        val_ood_loader = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=False, **kwargs)

    if val_ood_loader is None:
        raise ValueError(f"Unsupported out_dataset: {dataset[1]}")

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader":   val_ood_loader,
    })
