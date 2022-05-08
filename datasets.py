# Standard Library
import pickle
from typing import *
from pathlib import Path

# Third-party Party
import numpy as np
import PIL.Image as Image
from colorama import Fore, init

# Torch Library
import torch
import torch.utils.data as data

# My Library
from helper import visualize_np, visualize_plt, visualize_pil
from helper import ProjectPath, DatasetPath
from helper import ClassLabelLookuper

init(autoreset=True)

ImageType = TypeVar(
    "ImageType",
    np.ndarray, torch.Tensor, Path
)

ClassType = TypeVar(
    "ClassType",
    np.ndarray, torch.Tensor
)


class MultiDataset(data.Dataset):
    def __init__(self, dataset: str, split: str):
        super(MultiDataset, self).__init__()
        assert split in (s := ["train", "val", "test"]), f"{Fore.RED}Invalid split, s"
        self.split = split
        self.dataset = dataset
        self._dataset_reader: [str, Callable] = {
            "Cifar10": self.__read_cifar10,
            "Cifar100": self.__read_cifar100,
            "PascalVOC2012": self.__read_PascalVOC2012
        }
        assert dataset in self._dataset_reader.keys(), f"{Fore.RED}Invalid dataset, please select in " \
                                                       f"{self._dataset_reader.keys()}."
        self.image: Union[np.ndarray, List[np.ndarray]]
        self.label: np.ndarray
        self.image, self.label = self._dataset_reader[self.dataset]()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __read_cifar10(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.split in ["train", "val"]:
            data = []
            for batch in DatasetPath.Cifar10.train:
                with batch.open(mode="rb") as f:
                    data.append(pickle.load(f, encoding="bytes"))
            image = np.concatenate([i[b"data"].reshape(-1, 3, 32, 32) for i in data], axis=0)
            label = np.concatenate([i[b"labels"] for i in data], axis=0)
        else:
            with DatasetPath.Cifar10.test.open(mode="rb") as f:
                data = pickle.load(f, encoding="bytes")
            image = data[b"data"].reshape(-1, 3, 32, 32)
            label = data[b"labels"]
        return image, np.array(label)

    def __read_cifar100(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.split in ["train", "val"]:
            with DatasetPath.Cifar100.train.open(mode="rb") as f:
                data = pickle.load(f, encoding="bytes")
            image = data[b"data"].reshape(-1, 3, 32, 32)
            label = data[b"fine_labels"]
        else:
            with DatasetPath.Cifar100.test.open(mode="rb") as f:
                data = pickle.load(f, encoding="bytes")
            image = data["data"].reshape(-1, 3, 32, 32)
            label = data["label"]
        return image, np.asarray(label)

    def __read_PascalVOC2012(self) -> Tuple[List[Path], np.ndarray]:
        image = []
        label = []
        ccn = ClassLabelLookuper(datasets="PascalVOC2012")
        if self.split in "train":
            for k, v in DatasetPath.PascalVOC2012.train_idx.items():
                image.extend(v)
                label.extend([ccn.get_label(k)] * len(v))
        elif self.split == "val":
            for k, v in DatasetPath.PascalVOC2012.val_idx.items():
                image.extend(v)
                label.extend([ccn.get_label(k)] * len(v))
        else:
            assert False, f"{Fore.RED}PascalVOC2012 test data is not accesibly"
        image, idx = np.unique(image, return_index=True)
        return image, np.array(label)[idx]

if __name__ == "__main__":
    md = MultiDataset(dataset="PascalVOC2012", split="train")
    print("Done")
