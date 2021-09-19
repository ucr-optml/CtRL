import numpy as np
import os
import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

CLASS_NUM = 1623
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

class RandSplitOmniglot:
    def __init__(self, args):
        super(RandSplitOmniglot, self).__init__()
        data_root = os.path.join(args.data_dir, "omniglot")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.92206], std=[0.08426]
        )

        train_dataset = datasets.Omniglot(
            root=data_root,
            background=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset  = datasets.Omniglot(
            root=data_root,
            background=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        np.random.seed(args.seed_data)
        perm = np.random.permutation(CLASS_NUM)
        # print(perm)
        if args.ratio_samples == []:
            ratio = [1] * (CLASS_NUM//args.num_classes)
        elif len(args.ratio_samples) == CLASS_NUM//args.num_classes:
            ratio = args.ratio_samples
        else:
            ValueError('Ratio sample list error!')
        splits = [
            (
                partition_dataset(train_dataset, perm[args.num_classes * i : args.num_classes * (i + 1)], ratio[i]),
                partition_dataset(val_dataset, perm[args.num_classes * i : args.num_classes * (i + 1)]),
            )
            for i in range(CLASS_NUM//args.num_classes)
        ]

        
        [print(perm[args.num_classes * i : args.num_classes * (i + 1)]) for i in range(CLASS_NUM//args.num_classes)]

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

