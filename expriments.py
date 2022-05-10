import os
import sys
from typing import Optional
from helper import ProjectPath

def run(
        messgae: Optional[str] = None,
        log: Optional[bool] = True,
        dry_run: Optional[bool] = True,
        cifar: Optional[bool] = True,
        paper_train: Optional[bool] = True,
        paper_model: Optional[bool] = False,
        n_epoch: Optional[int] = 200,
        early_stop: Optional[int] = 50,
        log_loss_step: Optional[int] = 100,
        log_confusion_epoch: Optional[int] = 10,
        dataset: Optional[str] = "Cifar10",
    ) -> None:
    main_py = ProjectPath.base.joinpath("main.py")
    base_cmd = f"{sys.executable} {main_py} "

    # optional args
    if log:
        base_cmd += "-l "
    if dry_run:
        base_cmd += "-d "
    if cifar:
        base_cmd += "-c "
    if paper_train:
        base_cmd += "-pt "
    if paper_model:
        base_cmd += "-pm "
    
    base_cmd += f"-ne {n_epoch} "
    base_cmd += f"-es {early_stop} "
    base_cmd += f"-lls {log_loss_step} "
    base_cmd += f"-lce {log_confusion_epoch} "
    base_cmd += f"-ds {dataset} "

    print(s := f"Run command: {base_cmd}")

    if messgae is not None:
        base_cmd += f"-m \"{messgae}\" "
    else:
        base_cmd += f"-m \"{s}\""


    os.system(f"{base_cmd}")



if __name__ == "__main__":
    N_EPOCH = 200
    EARLY_STOP = 50

    # train
    run_idx = 0
    for dataset in ["Cifar10", "Cifar100", "PascalVOC2012"]:
        for paper_model in [False, True]:
            for paper_train in [True, False]:
                run_idx += 1
                run(
                    messgae=None,
                    n_epoch=N_EPOCH,
                    early_stop=EARLY_STOP,
                    log=True,
                    dry_run=False,
                    cifar=True if dataset[0] == "C" else False,
                    paper_train=paper_train,
                    paper_model=paper_model,
                    dataset=dataset
                )
