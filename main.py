import datetime

import tqdm
import numpy as np
from typing import *
from colorama import Fore, Style, init

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from dataset import Cifar10, Cifar100
from pathconfig import Path, PathConfig
from network import CifarAlexNet, AlexNet

init(autoreset=True)

class CifarTrainer:
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(self, network: Union[AlexNet, CifarAlexNet]):
        self.dtype = network.dtype
        self.device = self.available_device
        self.network = network.to(device=self.device)

        self.optim = optim.SGD(params=self.network.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
        self.loss_function = nn.CrossEntropyLoss()

        self.train_loader = data.DataLoader(c := Cifar100(split="train"), batch_size=128, shuffle=True, num_workers=1)
        self.validation_loader = data.DataLoader(Cifar100(split="val"), batch_size=128, shuffle=False, num_workers=1)
        self.test_loader = data.DataLoader(Cifar100(split="test"), batch_size=128, shuffle=False, num_workers=1)

        writer_path, self.checkpoint_path = self.decide_path(c)
        self.writer = SummaryWriter(log_dir=writer_path)

    def __del__(self):
        self.writer.close()

    @staticmethod
    def decide_path(dataset: data.Dataset) -> List[Path]:
        start_time = str(datetime.datetime.now())
        writer_path = PathConfig.runs / dataset.__name__ / start_time
        pt_path = PathConfig.check_point / dataset.__name__ / start_time / "best.pt"
        writer_path.mkdir(parents=True)
        pt_path.parent.mkdir(parents=True)
        return writer_path, pt_path

    def save_check_point(self):
        torch.save(self.network.state_dict(), f=self.checkpoint_path)

    def adjust_lr(self):
        self.optim.param_groups[0]["lr"] /= 10

    def train(self, n_epoch: int = 3000, early_stop: int = 100):
        x: torch.Tensor
        y: torch.Tensor
        y_pred: torch.Tensor
        loss: torch.Tensor

        max_acc = 0
        ealry_stop_cnt = 0
        before_stopping = 3
        for epoch in (tt := tqdm.trange(n_epoch)):
            # train
            self.network.train()
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(device=self.device, dtype=self.dtype), y.to(device=self.device, dtype=self.dtype)
                y_pred = self.network(x)
                train_loss = self.loss_function(y_pred.float(), y.long())
                train_loss.backward()
                self.optim.step()
                self.network.zero_grad()

                # log
                self.writer.add_scalar(tag="loss/train", scalar_value=train_loss.item(),
                                       global_step=epoch * len(self.train_loader) + step)

            # val
            val_acc = 0
            total_sum = 0
            self.network.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(self.validation_loader):
                    x, y = x.to(device=self.device, dtype=self.dtype), y.to(device=self.device, dtype=self.dtype)
                    y_pred = self.network(x)
                    val_loss = self.loss_function(y_pred.float(), y.long())

                    # log loss
                    self.writer.add_scalar(tag="loss/val", scalar_value=val_loss.item(),
                                           global_step=epoch * len(self.validation_loader) + step)
                    val_acc += torch.sum(y_pred.argmax(dim=1) == y)
                    total_sum += len(y)
            self.writer.add_scalar(tag="acc/val", scalar_value=(val_acc := (val_acc / total_sum)).item(),
                                   global_step=epoch)

            # test
            test_acc = 0
            total_sum = 0
            with torch.no_grad():
                for step, (x, y) in enumerate(self.test_loader):
                    x, y = x.to(device=self.device, dtype=self.dtype), y.to(device=self.device, dtype=self.dtype)
                    y_pred = self.network(x)
                    test_acc += torch.sum(y_pred.argmax(dim=1) == y)
                    total_sum += len(y)
            self.writer.add_scalar(tag="acc/test", scalar_value=(test_acc := (test_acc / total_sum)).item(),
                                   global_step=epoch)
            tt.write(
                f"Epoch [{epoch:>5d}|{n_epoch:>5d}], train loss: {train_loss:>7.6f}, val loss: {val_loss:>7.6f}, "
                f"val acc: {Fore.GREEN}{val_acc*100:>4.2f}%{Style.RESET_ALL}, max acc: {max_acc*100:>4.2f}%, test acc: {test_acc*100:>4.2f}%, "
                f"early stop: [{Fore.BLUE}{ealry_stop_cnt:>5d}{Style.RESET_ALL}|{early_stop:>5d}]"
            )

            if val_acc > max_acc:
                max_acc = val_acc
                ealry_stop_cnt = 0
                self.save_check_point()
            else:
                ealry_stop_cnt += 1

            # adjust lr as said in paper
            # if ealry_stop_cnt > 20 and before_stopping > 0:
            #     before_stopping -= 1
            #     early_stop_cn = 0
            #     self.adjust_lr()

            if ealry_stop_cnt > early_stop:
                tt.write(f"{Fore.RED}Early Stopped!")
                break
        return self



if __name__ == "__main__":
    ca = CifarAlexNet(in_size=1024, predict_class=256)
    ct = CifarTrainer(network=ca).train()
