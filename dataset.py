import re
import pickle
import pprint
from typing import *
from pathlib import Path

import dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from pathconfig import PathConfig


def gen_split(train_nums: int, train_ratio: float, path: Path):
    import random
    idx = list(range(train_nums))
    random.shuffle(idx)
    split = int(train_nums * train_ratio)
    train_idx, val_idx = np.array(idx[: split]), np.array(idx[split:])
    np.savez(path, train=train_idx, val=val_idx)
    return train_idx, val_idx


def load_split(path: Path):
    split = np.load(path)
    return split["train"], split["val"]


class Cifar10(Dataset):
    __name__ = "Cifar10"

    def __init__(self, split: str = "train"):
        assert split in ["train", "test", "val"]
        self.split = split
        self.name2label: Dict[str, int] = {}
        self.label2name: Dict[int, str] = {}
        self.data: Union[torch.Tensor, List[torch.Tensor]] = []
        self.label: Union[List[int], torch.Tensor] = []
        self.load()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def load(self):
        with PathConfig.Cifar10.cifar10_meta.open(mode="rb") as file:
            meta: dict = pickle.load(file)
            self.name2label = {name: i for i, name in enumerate(meta["label_names"])}
            self.label2name = {i: name for name, i in self.name2label.items()}
        name: str
        value: Path
        # load train or test
        for name, value in PathConfig.Cifar10.__dict__.items():
            if self.split != "test":
                if re.search("batch_\d", name) is not None:
                    with value.open(mode="rb") as file:
                        content: Dict[bytes, np.ndarray] = pickle.load(file, encoding="bytes")
                    image: torch.Tensor = torch.from_numpy(content[b"data"]).reshape(-1, 3, 32, 32).to(
                        torch.float) / 255
                    self.data.extend(image.unbind(dim=0))
                    self.label.extend(content[b"labels"])
            else:
                with PathConfig.Cifar10.cifar10_batch_test.open(mode="rb") as file:
                    content: Dict[bytes, np.ndarray] = pickle.load(file, encoding="bytes")
                image: torch.Tensor = torch.from_numpy(content[b"data"]).reshape(-1, 3, 32, 32)
                self.data.extend(image.unbind(dim=0))
                self.label.extend(content[b"labels"])
                self.data = torch.stack(self.data)
                self.label = torch.Tensor(self.label).long()
                return True

        # decide validation
        if (p := Path(__file__).resolve().parent.joinpath("cifar10_trainval.npz")).exists():
            train, val = load_split(path=p)
        else:
            train, val = gen_split(train_nums=len(self.data), train_ratio=0.8, path=p)

        if self.split == "val":
            idx = val
        else:
            idx = train
        self.data = torch.stack(self.data)[idx]
        self.label = torch.Tensor(self.label)[idx]

    def imshow(self, image: Union[None, torch.Tensor] = None, idx: Union[None, int] = None,
               title: Union[str, int] = None):
        assert not (image is None and idx is None), f"either image or idx should be determined"
        if image is not None:
            if image.ndim == 3:
                image = image.permute(1, 2, 0) if image.shape[0] == 3 else image
                if title is None:
                    title = ""
                else:
                    title = title if isinstance(title, str) else self.label2name[title]
                plt.imshow(image)
                plt.title(label="" if title is None else title)
        elif idx is not None:
            plt.imshow(self.data[idx].permute(1, 2, 0))
            plt.title(label=self.label2name[self.label[idx].item()])
        plt.show()


class Cifar100(Dataset):
    __name__ = "Cifar100"

    def __init__(self, split: str = "train"):
        assert split in ["train", "test", "val"]
        self.split = split
        self.name2label: Dict[str, int] = {}
        self.label2name: Dict[int, str] = {}
        self.data: Union[torch.Tensor, List[torch.Tensor]] = []
        self.label: Union[torch.Tensor, List[torch.Tensor]] = []
        self.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def load(self):
        with PathConfig.Cifar100.cifar100_meta.open(mode="rb") as file:
            meta: dict = pickle.load(file)
            self.name2label = {name: i for i, name in enumerate(meta["fine_label_names"])}
            self.label2name = {i: name for name, i in self.name2label.items()}
        if self.split != "test":
            with PathConfig.Cifar100.cifar100_train.open(mode="rb") as file:
                content: dict = pickle.load(file, encoding="bytes")
            image = torch.from_numpy(content[b"data"]).reshape(-1, 3, 32, 32).to(torch.float) / 255
            self.data.extend(image.unbind(dim=0))
            self.label.extend(content[b"fine_labels"])

            # decide data
            if (p := PathConfig.base.joinpath("cifar100_trainval.npz")).exists():
                train, val = load_split(path=p)
            else:
                train, val = gen_split(len(self.data), train_ratio=0.8, path=p)

            if self.split == "train":
                idx = train
            else:
                idx = val
            self.data = torch.stack(self.data)[idx]
            self.label = torch.Tensor(self.label)[idx].long()
        else:
            with PathConfig.Cifar100.cifar100_test.open(mode="rb") as file:
                content: Dict[bytes, np.ndarray] = pickle.load(file, encoding="bytes")
            image: torch.Tensor = torch.from_numpy(content[b"data"]).reshape(-1, 3, 32, 32) / 255
            self.data.extend(image.unbind(dim=0))
            self.label.extend(content[b"fine_labels"])
            self.data = torch.stack(self.data)
            self.label = torch.Tensor(self.label).long()
            return True

    def imshow(self, image: Union[None, torch.Tensor] = None, idx: Union[None, int] = None,
               title: Union[str, int] = None):
        assert not (image is None and idx is None), f"either image or idx should be determined"
        if image is not None:
            if image.ndim == 3:
                image = image.permute(1, 2, 0) if image.shape[0] == 3 else image
                if title is None:
                    title = ""
                else:
                    title = title if isinstance(title, str) else self.label2name[title]
                plt.imshow(image)
                plt.title(label="" if title is None else title)
        elif idx is not None:
            plt.imshow(self.data[idx].permute(1, 2, 0))
            plt.title(label=self.label2name[self.label[idx].item()])
        plt.show()


if __name__ == "__main__":
    import random

    # a = Cifar10(split="test")
    # for i in range(10):
    #     # a.imshow(random.choice(a.data))
    #     a.imshow(idx=i)
    # print(a.label2name)
    # for batch, label in (loader := DataLoader(dataset=a, batch_size=64, shuffle=True)):
    #     print(batch.shape)
    #     print(label.shape)
    #     break

    a = Cifar100(split="test")
    for i in range(10):
        a.imshow(idx=i)

    for batch, label in (loader := DataLoader(dataset=a, batch_size=128, shuffle=True)):
        print(batch.shape)
        print(label.shape)