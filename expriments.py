import os
from helper import ProjectPath

def runs(dataset: str, message: str):
    main = r"/usr/bin/env /home/jack/miniconda3/envs/torch/bin/python /home/jack/projects/AlexNet/main.py "
    main += "-l "
    main += f"-ds \"{dataset}\" "
    main += f"-m \"{message}\" "
    os.system(main)


if __name__ == "__main__":
    # runs(num_class=10, message=f"Task Length 10, Debug RUN")
    for run_idx in range(1, 6):
        for dataset in ["Cifar10", "Cifar100", "PascalVOC2012"]:
            runs(dataset=dataset, message=f"AlexNet on Dataset {dataset} Run {run_idx}")