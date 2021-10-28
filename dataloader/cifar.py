import numpy as np
import os
import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()
        data_root = os.path.join(args.data_dir, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset  = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        if args.ratio_samples == []:
            ratio = 1
        elif len(args.ratio_samples) == 1:
            ratio = args.ratio_samples[0]
        else:
            ValueError('Ratio sample list error!')
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(ratio * num_train))

        np.random.seed(args.seed_data)
        np.random.shuffle(indices)

        train_idx = indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
                ),
                torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size, **kwargs
                ),
            )
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader  = self.loaders[i][1]


def partition_dataset(dataset, perm, ratio=1.):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]
    newdataset.data = newdataset.data[: int(len(newdataset.data)*ratio)]
    newdataset.targets = [
        lperm.index(label)
        for label in newdataset.targets
        if label in lperm
    ]
    newdataset.targets = newdataset.targets[: int(len(newdataset.targets)*ratio)]

    return newdataset

class RandSplitCIFAR100:
    def __init__(self, args):
        super(RandSplitCIFAR100, self).__init__()
        data_root = os.path.join(args.data_dir, "cifar100")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset  = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        if args.seed_data < 0:
            perm = np.array(
                [[ 4, 30, 55, 72, 95],
                [ 1, 32, 67, 73, 91],
                [54, 62, 70, 82, 92],
                [ 9, 10, 16, 28, 61],
                [ 0, 51, 53, 57, 83],
                [22, 39, 40, 86, 87],
                [ 5, 20, 25, 84, 94],
                [ 6,  7, 14, 18, 24],
                [ 3, 42, 43, 88, 97],
                [12, 17, 37, 68, 76],
                [23, 33, 49, 60, 71],
                [15, 19, 21, 31, 38],
                [34, 63, 64, 66, 75],
                [26, 45, 77, 79, 99],
                [ 2, 11, 35, 46, 98],
                [27, 29, 44, 78, 93],
                [36, 50, 65, 74, 80],
                [47, 52, 56, 59, 96],
                [ 8, 13, 48, 58, 90],
                [41, 69, 81, 85, 89]]).reshape(-1)
            args.num_classes = 5
        else:
            np.random.seed(args.seed_data)
            perm = np.random.permutation(100)
            # perm = np.array(Superlabels).reshape(-1)
        print(perm)
        if args.ratio_samples == []:
            ratio = [1] * (100//args.num_classes)
        elif len(args.ratio_samples) == 100//args.num_classes:
            ratio = args.ratio_samples
        else:
            ValueError('Ratio sample list error!')
        splits = [
            (
                partition_dataset(train_dataset, perm[args.num_classes * i : args.num_classes * (i + 1)], ratio[i]),
                partition_dataset(val_dataset, perm[args.num_classes * i : args.num_classes * (i + 1)]),
            )
            for i in range(100//args.num_classes)
        ]

        
        [print(perm[args.num_classes * i : args.num_classes * (i + 1)]) for i in range(100//args.num_classes)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader  = self.loaders[i][1]

count = 0
