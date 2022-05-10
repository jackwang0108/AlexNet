import torch
import torch.nn as nn

from typing import *


# Attention: Since cifar image is only 32 * 32, ksize of first few convolutions layers must be modified,
# Attention：if you want to directly take in cifar image without resize

class CifarAlexNetPaper(nn.Module):
    def __init__(self, predict_class: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=(6, 6))
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.output = nn.Linear(in_features=4096, out_features=predict_class)

        for module in self.modules():
            if (attr := getattr(module, "weight", None)) is not None:
                self.dtype = attr.dtype

    def forward(self, x: torch.Tensor):
        # [batch, channel, width, height]
        x: torch.Tensor = self.conv1(x)
        x: torch.Tensor = self.conv2(x)
        x: torch.Tensor = self.conv3(x)
        x: torch.Tensor = self.conv4(x)
        x: torch.Tensor = self.conv5(x)
        x: torch.Tensor = self.avg_pool(x)
        x: torch.Tensor = x.flatten(start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = self.linear2(x)
        return self.output(x)


class AlexNetPaper(nn.Module):
    def __init__(self, predict_class: int, in_size: Optional[int] = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=(6, 6))
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.output = nn.Linear(in_features=4096, out_features=predict_class)

    def forward(self, x: torch.Tensor):
        # [batch, channel, width, height]
        x: torch.Tensor = self.conv1(x)
        x: torch.Tensor = self.conv2(x)
        x: torch.Tensor = self.conv3(x)
        x: torch.Tensor = self.conv4(x)
        x: torch.Tensor = self.conv5(x)
        x: torch.Tensor = self.avg_pool(x)
        x: torch.Tensor = x.flatten(start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = self.linear2(x)
        return self.output(x)


class AlexNet(nn.Module):
    def __init__(self, predict_class: int, in_size: Optional[int] = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.output = nn.Linear(in_features=4096, out_features=predict_class)

    def forward(self, x: torch.Tensor):
        # [batch, channel, width, height]
        x: torch.Tensor = self.conv1(x)
        x: torch.Tensor = self.conv2(x)
        x: torch.Tensor = self.conv3(x)
        x: torch.Tensor = self.conv4(x)
        x: torch.Tensor = self.conv5(x)
        x: torch.Tensor = self.avg_pool(x)
        x: torch.Tensor = x.flatten(start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = self.linear2(x)
        return self.output(x)


class CifarAlexNet(nn.Module):
    def __init__(self, in_size: Optional[int] = None, predict_class: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear1 = nn.Sequential(
            nn.Linear(256 * 6 * 6,
                      out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.output = nn.Linear(in_features=4096, out_features=predict_class)

        for module in self.modules():
            if (attr := getattr(module, "weight", None)) is not None:
                self.dtype = attr.dtype

    def forward(self, x: torch.Tensor):
        # [batch, channel, width, height]
        x: torch.Tensor = self.conv1(x)
        x: torch.Tensor = self.conv2(x)
        x: torch.Tensor = self.conv3(x)
        x: torch.Tensor = self.conv4(x)
        x: torch.Tensor = self.conv5(x)
        x: torch.Tensor = self.avg_pool(x)
        x: torch.Tensor = x.flatten(start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = self.linear2(x)
        return self.output(x)


if __name__ == "__main__":
    from datasets import MultiDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    T1 = T.Compose([
        T.Resize(size=(224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    md = MultiDataset(dataset="Cifar10", split="train").set_transform(T1)
    net = CifarAlexNet(predict_class=md.num_class)
    # net = AlexNet(predict_class=md.num_class)
    for x, y in DataLoader(md, batch_size=128):
        y_pred = net(x)
        print(y_pred.shape)
        break
