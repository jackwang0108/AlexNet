import torch
import torch.nn as nn


class CifarAlexNet(nn.Module):
    def __init__(self, in_size: int=1024, predict_class: int=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=4096),
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
            if (attr:=getattr(module, "weight", None)) is not None:
                self.dtype = attr.dtype

    def forward(self, x: torch.Tensor):
        # [batch, channel, width, height]
        x: torch.Tensor = self.conv1(x)
        x: torch.Tensor = self.conv2(x)
        x: torch.Tensor = self.conv3(x)
        x: torch.Tensor = self.conv4(x)
        x: torch.Tensor = self.conv5(x)
        x: torch.Tensor = x.flatten(start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = self.linear2(x)
        return self.output(x)

class AlexNet(nn.Module):
    def __init__(self, in_size: int, predict_class: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=4096),
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
        x: torch.Tensor = x.flatten(start_dim=1)
        x: torch.Tensor = self.linear1(x)
        x: torch.Tensor = self.linear2(x)
        return self.output(x)


if __name__ == "__main__":
    from dataset import Cifar10, Cifar100, DataLoader

    net = CifarAlexNet(in_size=1024, predict_class=10)
    # for x, y in DataLoader(Cifar10(split="train"), batch_size=128, shuffle=True):
    for x, y in DataLoader(Cifar100(split="train"), batch_size=128, shuffle=True):
        y_pred = net(x)
        break