import os
from typing import Optional
from helper import ProjectPath

def runs(dataset: str, message: str, extra: Optional[str] = None):
    main = r"/usr/bin/env /home/jack/miniconda3/envs/torch/bin/python /home/jack/projects/AlexNet/main.py "
    main += "-l "
    main += f"-ds \"{dataset}\" "
    main += f"-m \"{message}\" "
    if extra is not None:
        main += extra
    os.system(main)


if __name__ == "__main__":
    # for run_idx in range(1, 6):
    #     for dataset in ["Cifar10", "Cifar100", "PascalVOC2012"]:
    #         runs(dataset=dataset, message=f"AlexNet on Dataset {dataset} Run {run_idx}")
    runs(dataset="Cifar10", message=f"AlexNet on Cifar10, Run 1")
    runs(dataset="Cifar10", message=f"AlexNet on Cifar10 with paper setting, Run 1", extra="-p")
    runs(dataset="Cifar10", message=f"CifarAlexNet on Cifar10, Run 1", extra="-c")
    runs(dataset="Cifar10", message=f"CifarAlexNet on Cifar10 with paper setting, Run 1", extra="-c -p")

    runs(dataset="Cifar100", message=f"AlexNet on Cifar100, Run 1")
    runs(dataset="Cifar100", message=f"AlexNet on Cifar100 with paper setting, Run 1", extra="-p")
    runs(dataset="Cifar100", message=f"CifarAlexNet on Cifar100, Run 1", extra="-c")
    runs(dataset="Cifar100", message=f"CifarAlexNet on Cifar100 with paper setting, Run 1", extra="-c -p")

    runs(dataset="PascalVOC2012", message=f"AlexNet on PascalVOC2012, Run 1")
    runs(dataset="PascalVOC2012", message=f"AlexNet on PascalVOC2012 with paper setting, Run 1", extra="-p")