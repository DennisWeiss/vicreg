import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as F


class NominalCIFAR10ImageDataset(Dataset):
    def __init__(self, normal_class, transform, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=transform
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == normal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10ImageDataset(Dataset):
    def __init__(self, normal_class, transform, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=transform
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != normal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NormalCIFAR10DatasetRotationAugmented(Dataset):
    def __init__(self, normal_class, transform, train=True):
        self.normal_class = normal_class
        self.transform = transform

        data = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.data = [x[0] for x in data if x[1] == normal_class]

    def __len__(self):
        return 4 * len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx // 4].rotate(idx % 4 * 90))


class AnomalousCIFAR10DatasetRotationAugmented(Dataset):
    def __init__(self, normal_class, transform, train=True):
        self.normal_class = normal_class
        self.transform = transform

        data = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.data = [x[0] for x in data if x[1] != normal_class]

    def __len__(self):
        return 4 * len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx].rotate(idx % 4 * 90))