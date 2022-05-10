"""
Notes:
    心得:
        1. dataset定义处，如果要用torchvision，最好转为Image, 不然容易出错
        2. 网络定义处，最好写一个warmup来获得分类层这类和输入图像大小有关的数据
        3. 使用decrease learning rate的时候，需要加载最优的模型
        4. decrease learning rate的时候，平原的判定需要稳定一些，可以利用一个队列
        5. 使用logger的时候最好写一个hook决定是print还是log
        6. 写网络的时候多用用AvgPool2D
"""

# Standard Library
import re
import time
import logging
import argparse
import platform
import datetime
from typing import *
from pathlib import Path

# Third-Party Library
import numpy as np
from colorama import Fore, Style, init

# My Library
from network import AlexNetPaper, CifarAlexNetPaper
from network import AlexNet, CifarAlexNet
from datasets import MultiDataset
from helper import ProjectPath, DatasetPath
from helper import ClassificationEvaluator, ClassLabelLookuper
from helper import visualize_np, visualize_pil, visualize_plt

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T


init(autoreset=True)



class Trainer:
    start_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    avaliable_device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu:0")
    default_dtype: torch.dtype = torch.float

    num_worker: int = 0 if platform.system() == "Windows" else 2

    train_T = T.Compose([
        T.Resize(size=(256, 256)),
        T.RandomCrop(size=(224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(
        self, network: AlexNetPaper, dataset: str,
        log: Optional[bool] = True,
        dry_run: Optional[bool] = True,
        cifar: Optional[bool] = False,
        log_loss_step: Optional[int] = None,
        log_confusion_epoch: Optional[int] = None
    ) -> None:

        # save param
        self.log = log
        self.dataset: str = dataset
        self.dry_run: bool = dry_run
        self.cifar: bool = cifar
        self.log_loss_step: Union[None, int] = log_loss_step
        self.log_confusion_epoch: Union[None, int] = log_confusion_epoch
        self.network: AlexNetPaper = network.to(
            self.avaliable_device, non_blocking=True)

        # make file suffix
        suffix = self.dataset + "/" + self.network.__class__.__name__

        # dataset
        # remove transform first
        if self.cifar:
            self.train_T.transforms = self.train_T.transforms[2:]
        self.train_ds = MultiDataset(dataset=self.dataset, split="train").set_transform(
            self.train_T
        )
        self.val_ds = MultiDataset(dataset=self.dataset, split="val").set_transform(
            self.train_T
        )

        # log
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter(fmt="%(message)s", datefmt="")

        terminal = logging.StreamHandler()
        terminal.setLevel(logging.INFO)
        terminal.setFormatter(fmt=fmt)
        self.logger.addHandler(terminal)

        if log:
            self.log_path = ProjectPath.log / suffix / f"{self.start_time}.log"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"{Fore.YELLOW}Save log to {self.log_path.relative_to(ProjectPath.base)}")
            file = logging.FileHandler(str(self.log_path))
            file.setLevel(logging.INFO)
            file.setFormatter(fmt)
            self.logger.addHandler(file)

        # Tensorboard
        if not dry_run:
            writer_path = ProjectPath.runs / suffix / self.start_time
            self.logger.info(
                f"{Fore.YELLOW}Training curves can be found in {writer_path.relative_to(ProjectPath.base)}")
            self.writer = SummaryWriter(log_dir=writer_path)

        # checkpoints
        if not dry_run:
            self.checkpoint_path = ProjectPath.checkpoints / suffix / self.start_time / "best.pt"
            self.logger.info(
                f"{Fore.YELLOW}Save checkpoint to {self.checkpoint_path.relative_to(ProjectPath.base)}")
            self.checkpoint_path.parent.mkdir(parents=True)

    def __del__(self):
        if not self.dry_run:
            self.writer.close()

        if self.log:
            self.logger.info(f"{Fore.MAGENTA}Shutdown at {datetime.datetime.now()}, waiting...")
        else:
            print(f"{Fore.MAGENTA}Shutdown at {datetime.datetime.now()}, waiting...")
        # shutdown logging and flush buffer
        # wait for write to file
        time.sleep(3)
        logging.shutdown()

        # clean logs
        if self.log:
            with self.log_path.open(mode="r+") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    lines[i] = re.sub(r"..\d\dm", "", lines[i])
                f.seek(0)
                f.write("".join(lines[:-1]))

    def modern_train(
        self, 
        lr: Optional[float] = 1e-3,
        n_epoch: Optional[int] = 200,
        early_stop: Optional[int] = 30,
        message: Optional[str] = None
    ) -> AlexNetPaper:
        # log training digest
        self.logger.info(f"Training Digest: {message}")
        self.logger.info(f"AlexNet training with modern setup")
        self.logger.info(f"lr: {lr}")
        self.logger.info(f"e_poech: {n_epoch}")
        self.logger.info(f"early_stop: {early_stop}")
        self.logger.info(f"datasets: {self.train_ds.dataset}")
        

        # detect anomaly
        torch.autograd.set_detect_anomaly(True)

        # train widgets
        train_evaluator = ClassificationEvaluator(dataset=self.train_ds.dataset)
        val_evaluator = ClassificationEvaluator(dataset=self.val_ds.dataset)
        train_loader = data.DataLoader(
            self.train_ds, batch_size=128, shuffle=True, num_workers=self.num_worker,
            pin_memory=True
        )
        val_loader = data.DataLoader(
            self.val_ds, batch_size=128, shuffle=False, num_workers=self.num_worker,
            pin_memory=True
        )
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.network.parameters())
        self.logger.info(f"Optim: {optimizer.__class__.__name__}")
        optimizer.zero_grad()

        # constant
        max_top1: float = 0
        early_stop_cnt: int = 0
        ne_digits: int = len(str(n_epoch))
        es_digits: int = len(str(early_stop))

        # typing
        x: torch.Tensor
        y: torch.Tensor
        loss: torch.Tensor

        # train network
        for epoch in range(100000 if n_epoch is None else n_epoch):
            # setup evaluator
            train_evaluator.new_epoch()
            val_evaluator.new_epoch()

            # train
            self.network.train()
            for step, (x, y) in enumerate(train_loader):
                x = x.to(device=self.avaliable_device, dtype=self.default_dtype, non_blocking=True)
                y = y.to(device=self.avaliable_device, dtype=torch.long, non_blocking=True)

                # inference
                y_pred = self.network(x)

                # gradient descent
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # log
                if self.log and (self.log_loss_step is not None and step % self.log_loss_step == 0):
                    self.logger.info(f"{Fore.GREEN}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}d}|{n_epoch}], step: {step}, loss: {loss:>.5f}")
                if not self.dry_run:
                    if step % self.log_loss_step == 0:
                        self.writer.add_scalar(tag="train-loss", scalar_value=loss.cpu().item(), global_step=step + len(train_loader) * epoch)
                train_evaluator.record(y_pred=y_pred, y=y)

            # val
            self.network.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(val_loader):
                    x = x.to(device=self.avaliable_device, dtype=self.default_dtype, non_blocking=True)
                    y = y.to(device=self.avaliable_device, dtype=torch.long)

                    # inference
                    y_pred = self.network(x)

                    # log
                    val_evaluator.record(y_pred=y_pred, y=y)
            
            # early stop
            new_acc = val_evaluator.acc
            if max_top1 <= new_acc[0]:
                early_stop_cnt = 0
                max_top1 = new_acc[0]
                self.logger.info(
                    f"{Fore.BLUE}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}}|{n_epoch}], "\
                    f"new top1 Acc: {new_acc[0]:>.5f}, top5 Acc:{new_acc[1]:>.5f}"
                )
                if not self.dry_run:
                    torch.save(self.network.state_dict(), self.checkpoint_path)
                    self.logger.info(
                        f"{Fore.MAGENTA}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}}|{n_epoch}], "\
                        f"save checkpoint to {self.checkpoint_path.relative_to(ProjectPath.base)}"
                    )
            else:
                early_stop_cnt += 1
                self.logger.info(
                    f"{Fore.BLUE}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}}|{n_epoch}], "\
                    f"top1 Acc: [{new_acc[0]:>5f}|{max_top1:>.5f}], top5 Acc: [{new_acc[1]:>5f}] early_stop_cnt: [{early_stop_cnt:>{es_digits}d}|{early_stop}]"
                )

            # tensorboard
            if not self.dry_run:
                self.writer.add_scalars(
                    main_tag=f"Train Accuracy",
                    tag_scalar_dict={
                        "mAcc-top1": new_acc[0],
                        "mAcc-top5": new_acc[1],
                        "max-mAcc-top1": max_top1
                    },
                    global_step=epoch
                )
            
            # print confusion matrix
            # if self.log_confusion_epoch is not None and epoch % self.log_confusion_epoch == 0:
            #     table = val_evaluator.get_confusion(top=5, title=f"Top 5 Confusion Matrix of dataset {self.dataset}")
            #     if self.log:
            #         self.logger.info(str(table))
            #     else:
            #         print(table)

        return self.network


    def paper_train(
        self,
        n_epoch: Optional[int] = 200,
        early_stop: Optional[int] = 30,
        message: Optional[str] = None
    ) -> AlexNetPaper:
        # log training digest
        self.logger.info("Start Training".center(200, "+"))
        self.logger.info(f"Training Digest: {message}")
        self.logger.info(f"AlexNet training with paper setup")
        self.logger.info(f"e_poech: {n_epoch}")
        self.logger.info(f"early_stop: {early_stop}")
        self.logger.info(f"datasets: {self.train_ds.dataset}")
        

        # detect anomaly
        torch.autograd.set_detect_anomaly(True)

        # train widgets
        train_evaluator = ClassificationEvaluator(dataset=self.train_ds.dataset)
        val_evaluator = ClassificationEvaluator(dataset=self.val_ds.dataset)
        train_loader = data.DataLoader(
            self.train_ds, batch_size=128, shuffle=True, num_workers=self.num_worker,
            pin_memory=True
        )
        val_loader = data.DataLoader(
            self.val_ds, batch_size=128, shuffle=False, num_workers=self.num_worker,
            pin_memory=True
        )
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=self.network.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        optimizer.zero_grad()

        # constant
        max_top1: float = 0
        max_top5: float = 0
        plateau: int = int(early_stop * 1/3)
        plateau_cnt: int = 0
        before_stop: int = 3
        early_stop_cnt: int = 0
        ne_digits: int = len(str(n_epoch))
        es_digits: int = len(str(early_stop))
        p_digits: int = len(str(plateau))

        # typing
        x: torch.Tensor
        y: torch.Tensor
        loss: torch.Tensor

        # train network
        last_best_epoch: List[int] = []
        for epoch in range(100000 if n_epoch is None else n_epoch):
            # setup evaluator
            train_evaluator.new_epoch()
            val_evaluator.new_epoch()

            # train
            self.network.train()
            for step, (x, y) in enumerate(train_loader):
                x = x.to(device=self.avaliable_device, dtype=self.default_dtype, non_blocking=True)
                y = y.to(device=self.avaliable_device, dtype=torch.long)

                # inference
                y_pred = self.network(x)

                # gradient descent
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # log
                if self.log and (self.log_loss_step is not None and step % self.log_loss_step == 0):
                    self.logger.info(f"{Fore.GREEN}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}d}|{n_epoch}], step: {step}, loss: {loss:>.5f}")
                if not self.dry_run:
                    if self.log_loss_step is not None and step % self.log_loss_step == 0:
                        self.writer.add_scalar(tag="train-loss", scalar_value=loss.cpu().item(), global_step=step + len(train_loader) * epoch)
                train_evaluator.record(y_pred=y_pred, y=y)

            # val
            self.network.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(val_loader):
                    x = x.to(device=self.avaliable_device, dtype=self.default_dtype, non_blocking=True)
                    y = y.to(device=self.avaliable_device, dtype=torch.long, non_blocking=True)

                    # inference
                    y_pred = self.network(x)

                    # log
                    val_evaluator.record(y_pred=y_pred, y=y)
            
            # early stop
            new_acc = val_evaluator.acc
            if max_top1 <= new_acc[0]:
                early_stop_cnt = 0
                max_top1 = new_acc[0]
                max_top5 = new_acc[1]
                last_best_epoch.append(epoch)
                self.logger.info(
                    f"{Fore.BLUE}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}}|{n_epoch}], "\
                    f"new top1 Acc: {new_acc[0]:>.5f}, new top5 Acc:{new_acc[1]:>.5f}, "\
                    f"lr: {optimizer.param_groups[0]['lr']}, "\
                    f"Plateau: [{plateau_cnt:>{p_digits}d}|{plateau}]"
                )
                if not self.dry_run:
                    torch.save(self.network.state_dict(), self.checkpoint_path)
                    self.logger.info(
                        f"{Fore.MAGENTA}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}}|{n_epoch}], "\
                        f"save checkpoint to {self.checkpoint_path.relative_to(ProjectPath.base)}"
                    )
            else:
                early_stop_cnt += 1
                self.logger.info(
                    f"{Fore.BLUE}Dataset: {self.dataset}, Epoch: [{epoch:>{ne_digits}}|{n_epoch}], "\
                    f"top1 Acc: [{new_acc[0]:>5f}|{max_top1:>.5f}], top5 Acc: [{new_acc[1]:>.5f}|{max_top5:>.5f}], "\
                    f"early_stop_cnt: [{early_stop_cnt:>{es_digits}d}|{early_stop}], "\
                    f"lr: {optimizer.param_groups[0]['lr']}, "\
                    f"Plateau: [{plateau_cnt:>{p_digits}d}|{plateau}]"
                )

            # adjust lr as said in paper
            if len(last_best_epoch) > 4 and (plateau_cnt := epoch - last_best_epoch[-2]) >= plateau and before_stop > 0:
                before_stop -= 1
                early_stop_cnt = 0
                for param_group in optimizer.param_groups:
                    param_group["lr"] /= 10
                self.network.load_state_dict(torch.load(self.checkpoint_path, map_location=self.avaliable_device))
                before = optimizer.param_groups[0]["lr"] * 10
                self.logger.info(f"{Fore.YELLOW}Decrease lr at epoch: {epoch}, from {before_stop} to {before / 10}, switch to best model")

            # tensorboard
            if not self.dry_run:
                self.writer.add_scalars(
                    main_tag=f"Train Accuracy",
                    tag_scalar_dict={
                        "mAcc-top1": new_acc[0],
                        "mAcc-top5": new_acc[1],
                        "max-mAcc-top1": max_top1
                    },
                    global_step=epoch
                )

            # early stop
            if early_stop_cnt >= early_stop:
                self.logger.info(f"{Fore.MAGENTA}Early Stopped at epoch: {epoch}")
            # print confusion matrix
            # if self.log_confusion_epoch is not None and epoch % self.log_confusion_epoch == 0:
            #     table = val_evaluator.get_confusion(top=5, title=f"Top 5 Confusion Matrix of dataset {self.dataset}")
            #     if self.log:
            #         self.logger.info(str(table))
            #     else:
            #         print(table)

        return self.network

def parse_arg() -> argparse.Namespace:
    def green(s): return f"{Fore.GREEN}{s}{Style.RESET_ALL}"
    def yellow(s): return f"{Fore.YELLOW}{s}{Style.RESET_ALL}"
    def blue(s): return f"{Fore.BLUE}{Style.BRIGHT}{s}{Style.RESET_ALL}"

    parser = argparse.ArgumentParser(description=blue("AlexNet Pytorch Implementation training util by Shihong Wang (Jack3Shihong@gmail.com)"))
    parser.add_argument("-v", "--version", action="version", version="%(prog)s v2.0, fixed training bugs, but there's still GPU memory leak problem")
    parser.add_argument("-d", "--dry_run", dest="dry_run", default=False, action="store_true", help=green("If run without saving tensorboard amd network params to runs and checkpoints"))
    parser.add_argument("-l", "--log", dest="log", default=False, action="store_true", help=green("If save terminal output to log"))
    parser.add_argument("-c", "--cifar", dest="cifar", default=False, action="store_true", help=green("If use cifar modified network"))
    parser.add_argument("-p", "--paper", dest="paper", default=False, action="store_true", help=green("If train the network using paper setting"))
    parser.add_argument("-ne", "--n_epoch", dest="n_epoch", type=Union[int, None], default=250, help=yellow("Set maximum training epoch of each task"))
    parser.add_argument("-es", "--early_stop", dest="early_stop", type=int, default=50, help=yellow("Set maximum early stop epoch counts"))
    parser.add_argument("-lls", "--log_loss_step", dest="log_loss_step", type=Union[int, None], default=100, help=yellow("Set log loss steps"))
    parser.add_argument("-lce", "--log_confusion_epoch", dest="log_confusion_epoch", type=int, default=10, help=yellow("Set log confusion matrix epochs"))
    parser.add_argument("-ds", "--dataset", dest="dataset", type=str, default="Cifar10", help=blue("Set training datasets"))
    parser.add_argument("-m", "--message", dest="message", type=str, default=f"", help=blue("Training digest"))
    return parser.parse_args()


if __name__ == "__main__":
    # get arg
    args = parse_arg()
    
    # Attention: Parameters
    log: bool = args.log
    dry_run: bool = args.dry_run
    cifar: bool = args.cifar
    paper: bool = args.paper
    n_epoch: int = args.n_epoch
    early_stop: int = args.early_stop
    log_loss_step: int = args.log_loss_step
    log_confusion_epoch: int = args.log_confusion_epoch
    messgae: str = args.message
    dataset: str = args.dataset

    assert dataset in (s:=["Cifar10", "Cifar100", "PascalVOC2012"]), f"{Fore.RED}Invalid Datasets, please select in {s}"

    if cifar:
        network = CifarAlexNetPaper(predict_class=len(ClassLabelLookuper(datasets=dataset).cls))
    else:
        network = AlexNetPaper(predict_class=len(ClassLabelLookuper(datasets=dataset).cls))
    
    trainer = Trainer(
        network=network, dataset=dataset, log=log, dry_run=dry_run,
        cifar=cifar,
        log_loss_step=log_loss_step,
        log_confusion_epoch=log_confusion_epoch
    )

    if paper:
        network = trainer.paper_train(
            n_epoch=n_epoch, early_stop=early_stop,
            message=messgae
        )
    else:
        network = trainer.modern_train(
            lr=1e-4, n_epoch=n_epoch, early_stop=early_stop,
            message=messgae
        )
