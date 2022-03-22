from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)


class PathConfig:
    # base
    base: Path = Path(__file__).resolve().parent
    dataset: Path = base.joinpath("dataset")
    runs: Path = base.joinpath("runs")
    check_point: Path = base.joinpath("checkpoints")

    # cifar 10
    class Cifar10:
        base: Path = Path(__file__).resolve().parent
        dataset: Path = base.joinpath("dataset")
        cifar10: Path = dataset.joinpath("cifar-10").resolve()
        cifar10_meta = cifar10.joinpath("batches.meta")
        cifar10_batch_1: Path = cifar10.joinpath("data_batch_1")
        cifar10_batch_2: Path = cifar10.joinpath("data_batch_2")
        cifar10_batch_3: Path = cifar10.joinpath("data_batch_3")
        cifar10_batch_4: Path = cifar10.joinpath("data_batch_4")
        cifar10_batch_5: Path = cifar10.joinpath("data_batch_5")
        cifar10_batch_test: Path = cifar10.joinpath("test_batch")

    class Cifar100:
        base: Path = Path(__file__).resolve().parent
        dataset: Path = base.joinpath("dataset")
        cifar100: Path = dataset.joinpath("cifar-100").resolve()
        cifar100_meta: Path = cifar100.joinpath("meta")
        cifar100_test: Path = cifar100.joinpath("test")
        cifar100_train: Path = cifar100.joinpath("train")


if __name__ == "__main__":
    name: str
    value: Path
    for name, value in PathConfig.__dict__.items():
        if isinstance(value, Path):
            print(f"{name}: {value}, {Fore.GREEN}{value.exists()}")
        elif (base := getattr(value, "base", None)) is not None and isinstance(base, Path):
            print(f"{Fore.GREEN}{value.__name__}")
            for [inner_name, inner_value] in value.__dict__.items():
                if isinstance(inner_value, Path):
                    print(f"{inner_name}: {inner_value}, {Fore.GREEN}{inner_value.exists()}")
