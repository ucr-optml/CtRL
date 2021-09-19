import os
import torch
from torchvision import datasets, transforms

import numpy as np


class Permute(object):
    def __call__(self, tensor):
        out = tensor.flatten()
        out = out[self.perm]
        return out.view(1, 28, 28)

    def __repr__(self):
        return self.__class__.__name__

class PermutedMNIST:
    def __init__(self, args):
        super(PermutedMNIST, self).__init__()
        self.args = args
        data_root = os.path.join(args.data_dir, "mnist")

        use_cuda = torch.cuda.is_available()

        self.permuter = Permute()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    self.permuter,
                ]
            ),
        )
        val_dataset = datasets.MNIST(
            data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    self.permuter,
                ]
            ),
        )
        np.random.seed(args.seed_data)
        self.seed = np.random.randint(1000000)
        print(self.seed)
        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

    def update_task(self, i):
        np.random.seed(i + self.seed)
        self.permuter.__setattr__("perm", np.random.permutation(784))



count = 0

class Rotate(object):
    def __call__(self, img):
        out = transforms.functional.rotate(img, self.angle)
        return out

    def __repr__(self):
        return self.__class__.__name__


class RotatedMNIST:
    def __init__(self, args):
        global count
        self.args = args
        super(RotatedMNIST, self).__init__()

        data_root = os.path.join(args.data_dir, "mnist")

        use_cuda = torch.cuda.is_available()

        self.rotater = Rotate()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(3),
                    self.rotater,
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        val_dataset = datasets.MNIST(
            data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(3),
                    self.rotater,
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        np.random.seed(args.seed_data)
        self.perm = np.random.permutation(args.num_tasks)
        print(self.perm)
        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

    def update_task(self, i):
        self.rotater.__setattr__('angle', int(self.perm[i])*(360 // self.args.num_tasks))

        