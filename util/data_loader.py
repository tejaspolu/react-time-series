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
        self.pred  = data['prediction'][::skip]
        self.truth = data['Truth'][::skip]

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, i):
        return (torch.from_numpy(self.pred[i]).float(),
                torch.from_numpy(self.truth[i]).float())


def get_loader_in(args, config_type='default', split=('train', 'val')):
    # … your existing get_loader_in code unchanged …
    # (includes RK4Dataset branch for in_dataset='rk4')
    ...


def get_loader_out(args, dataset=(''), config_type='default', split=('train', 'val')):
    """
    dataset: tuple (in_dataset, out_dataset).  E.g. ('rk4','noise_series')
    """
    config = EasyDict({
        "default": {
            'transform_train':        transform_train,
            'transform_test':         transform_test,
            'transform_train_largescale': transform_train_largescale,
            'transform_test_largescale':  transform_test_largescale,
            'batch_size':             args.batch_size
        },
    })[config_type]

    train_ood_loader = None
    val_ood_loader   = None

    # ─── existing image OOD branches ────────────────────────────────────
    # e.g. SVHN, DTD, CIFAR-100, etc...
    # (All of your original code for those stays here.)

    # ─── new: Gaussian‐noise time‐series for 'noise_series' ─────────────
    if 'val' in split and dataset[1].lower() == 'noise_series':
        # load ID .npy to get (T, L)
        data = np.load(args.input_file, allow_pickle=True).item()
        T, L = data['prediction'].shape

        class NoiseSeriesDataset(Dataset):
            def __init__(self, T, L):
                self.T = T
                self.L = L
            def __len__(self):
                return self.T
            def __getitem__(self, idx):
                # random normal noise matching length L
                x = np.random.randn(self.L).astype(np.float32)
                y = np.zeros(self.L, dtype=np.float32)  # dummy label
                return (torch.from_numpy(x), torch.from_numpy(y))

        ds = NoiseSeriesDataset(T, L)
        val_ood_loader = DataLoader(ds,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    **kwargs)

    # ─── fallback: if nothing matched, error out ─────────────────────────
    if val_ood_loader is None:
        raise ValueError(f"Unsupported out_dataset: {dataset[1]}")

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader":   val_ood_loader,
    })
