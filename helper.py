# Standard Library
import pickle
from typing import *
from pathlib import Path
from dataclasses import dataclass

# Third-Party Library
import torch
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

init(autoreset=True)


@dataclass
class ProjectPath:
    base: Path = Path(__file__).resolve().parent
    runs: Path = base.joinpath("runs")
    config: Path = base.joinpath("config")
    dataset: Path = base.joinpath("dataset")
    checkpoints: Path = base.joinpath("checkpoints")

    def __init__(self) -> None:
        for project_path in ProjectPath.__dict__.values():
            if isinstance(project_path, Path):
                project_path.mkdir(parents=True, exist_ok=True)


class DatasetPath:
    base: Path = ProjectPath.dataset

    class Cifar10:
        base: Path = ProjectPath.base.joinpath("dataset/cifar-10")
        meta: Path = base.joinpath("batches.meta")
        test: Path = base.joinpath("test_batch")
        train: List[Path] = list(base.glob("data*"))

    class Cifar100:
        base: Path = ProjectPath.base.joinpath("dataset/cifar-100")
        meta: Path = base.joinpath("meta")
        test: Path = base.joinpath("test")
        train: Path = base.joinpath("train")

    class PascalVOC2012:
        base: Path = ProjectPath.base.joinpath("dataset/PascalVOC2012")

        _train_class_idx: List[Path] = base.joinpath("ImageSets", "Main").glob(r"*_train.txt")
        _val_class_idx: List[Path] = base.joinpath("ImageSets", "Main").glob(r"*_val.txt")

        # get train
        train_idx: Dict[str, List[Path]] = {}
        for path in _train_class_idx:
            cls = path.stem.split("_")[0]
            train_idx[cls] = []
            with path.open(mode="r") as f:
                c = f.readlines()
            for line in c:
                # train_idx[cls].
                if line[-3:-1] == "-1":
                    continue
                train_idx[cls].append(
                    base.joinpath("JEPGImages", f"{line.split(' ')[0]}.jpg")
                )

        # get validation
        val_idx: Dict[str, List[Path]] = {}
        for path in _val_class_idx:
            cls = path.stem.split("_")[0]
            val_idx[cls] = []
            with path.open(mode="r") as f:
                c = f.readlines()
            for line in c:
                # train_idx[cls].
                if line[-3:-1] == "-1":
                    continue
                val_idx[cls].append(
                    base.joinpath("JEPGImages", f"{line.split(' ')[0]}.jpg")
                )


class ClassLabelLookuper:
    def __init__(self, datasets: str) -> None:
        assert datasets in (s := [name for name, value in DatasetPath.__dict__.items() if isinstance(value, type)]), \
            f"{Fore.RED}Invalid Dataset, should be in {s}, but you offered {datasets}"

        self.cls: List[str]
        self._cls2label: Dict[str, int]
        self._label2cls: Dict[int, str]

        if datasets == "Cifar10":
            with DatasetPath.Cifar10.meta.open(mode="rb") as f:
                meta = pickle.load(f)
            self.cls = meta["label_names"]
        elif datasets == "Cifar100":
            with DatasetPath.Cifar100.meta.open(mode="rb") as f:
                meta = pickle.load(f)
            self.cls = meta["fine_label_names"]
        else:
            self.cls = DatasetPath.PascalVOC2012.train_idx.keys()

        self._cls2label = dict(zip(self.cls, range(len(self.cls))))
        self._label2cls = dict(zip(range(len(self.cls)), self.cls))

    def get_class(self, label: int) -> str:
        return self._label2cls[label]

    def get_label(self, cls: str) -> int:
        return self._cls2label[cls]


ImageType = TypeVar(
    "ImageType",
    np.ndarray, torch.Tensor, Image.Image,
    List[np.ndarray], List[torch.Tensor], List[Image.Image]
)

ClassType = TypeVar(
    "ClassType",
    str,
    List[np.ndarray], List[torch.Tensor], List[Image.Image]
)


def _get_image(return_png: bool = False, driver: str = "ndarray"):
    assert driver in ["pil", "ndarray"]

    def visualize_func_decider(show_func: Callable = _visualize) -> Callable:
        def show_with_png(*args, **kwargs):
            show_func(*args, **kwargs)
            import matplotlib.backends.backend_agg as bagg
            canvas = bagg.FigureCanvasAgg(plt.gcf())
            canvas.draw()
            png, (width, height) = canvas.print_to_buffer()
            png = np.frombuffer(png, dtype=np.uint8).reshape(
                (height, width, 4))

            if driver == 'pil':
                return Image.fromarray(png)
            else:
                return png

        if return_png:
            return show_with_png
        else:
            return show_func

    return visualize_func_decider


def _visualize(image: ImageType, cls: Optional[ClassType] = None) -> None:
    image_list: List[np.ndarray]
    title_list: List[str]

    # type check
    assert isinstance(
        image, (Image.Image, np.ndarray, torch.Tensor, list)
    ), f"{Fore.RED}Wrong type, input type of image should be (Image.Image, np.ndarray, torch.Tensor, list). " \
       f"But received {type(image)}"
    if isinstance(image, list):
        assert all(
            isinstance(i, (Image.Image, np.ndarray, torch.Tensor, list)) for i in image
        ), f"{Fore.RED}Wrong type, input image type in the list should be (Image.Image, np.ndarray, torch.Tensor, " \
           f"list). But not all image in the image are valid"

    assert isinstance(cls,
                      (str, list)) or cls is None, f"{Fore.RED}Wrong type, input type of cls should be (str, list). " \
                                                   f"But received {type(cls)}"
    if isinstance(cls, list):
        assert all(
            isinstance(i, str) for i in cls
        ), f"{Fore.RED}Worng type, input cls in the list type should be str, but not all cls in the cls are valid"

    # make image
    image_list = []
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if isinstance(image, (np.ndarray, torch.Tensor)):
        assert image.ndim in [2, 3, 4], \
            f"{Fore.RED}Wrong shape, input dimension should be " \
            f"2: [height, width] for single 1-channel gray image, " \
            f"3: [height, width, channel] for single 3-channel color image, or multiple gray image and " \
            f"4: [batch, height, width, channel] for multiple 3-channel color image"
        image = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
        # gray image
        if image.ndim == 2:
            image_list.append(np.expand_dims(image, axis=-1))
        elif image.ndim == 3:
            if (c_num := image.shape[-1]) == 3:
                # single 3-channel colored image [height, width, channel]
                image_list.append(image)
            elif (c_num := image.shape[0]) == 3:
                # single 3-channel colored image [channel, height, width]
                image_list.append(image.transpose(1, 2, 0))
            else:
                # multiple 1-channel gray image
                image_list.extend([image[..., i] for i in range(c_num)])
        else:
            # multiple 3-channel color image
            if image.shape[1] == 3:
                image = image.transpose(0, 2, 3, 1)
            image_list.extend([image[i, ...] for i in range(image.shape[0])])
    else:
        for img in image:
            if isinstance(img, Image.Image):
                image = np.asarray(img)
            if isinstance(img, (np.ndarray, torch.Tensor)):
                assert img.ndim in [2, 3], f"{Fore.RED}Wrong shape, input dimension should be " \
                                           f"2: [height, width] for single 1-channel gray image, " \
                                           f"3: [height, width, channel] for 3-channel color image"
                img = img if isinstance(img, np.ndarray) else img.detach().cpu().numpy()
                if img.ndim == 2:
                    image_list.append(np.expand_dims(img, axis=-1))
                elif img.ndim == 3:
                    assert img.shape[-1] == 3, f"{Fore.RED}Wrong shape, input image in the list should be 3-channel " \
                                               f"image"
                    image_list.append(img)

    # make title
    num_image = len(image_list)
    title_list = cls if isinstance(cls, list) else ["" if cls is None else cls] * num_image
    assert (num_cls := len(title_list)) == num_image or not isinstance(cls, list), \
        f"{Fore.RED}Image num and class num mismatch, {num_cls} images with {num_image} class"

    # draw
    import math
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

    n_row: int = int(math.sqrt(num_image))
    n_col: int = math.ceil(num_image / n_row)

    ax: List[List[Axes]]
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, layout="tight", figsize=(2 * n_row, 2 * n_col))
    if len(image_list) == 1:
        ax = [[ax]]
    elif n_row == 1:
        ax = [ax]

    fill_num = n_col * n_row - len(image_list)
    image_list.extend([np.ones(shape=(10, 10, 3), dtype=int) * 255] * fill_num)
    title_list.extend([""] * fill_num)

    for img_idx, (img, title) in enumerate(zip(image_list, title_list)):
        row = img_idx // n_col
        col = img_idx % n_col
        ax[row][col].imshow(img)
        ax[row][col].set_title(title)
        ax[row][col].set_axis_off()

    canvas = Canvas(fig)
    canvas.draw()


def visualize_plt(*args, **kwargs) -> None:
    _visualize(*args, **kwargs)
    plt.ion()
    plt.show()


def visualize_np(*args, **kwargs) -> np.ndarray:
    img = _get_image(return_png=True, driver="ndarray")(_visualize)(*args, **kwargs)
    plt.ion()
    plt.show()
    return img


def visualize_pil(*args, **kwargs) -> Image.Image:
    return _get_image(return_png=True, driver="pil")(_visualize)(*args, **kwargs)


if __name__ == "__main__":
    import pprint

    # pp = ProjectPath()
    # print(DatasetPath.Cifar10.train)
    # print(DatasetPath.Cifar100.train)
    # print(DatasetPath.PascalVOC2012.train_idx.keys().__len__())
    # print(type(DatasetPath), isinstance(DatasetPath, type))
    # print([name for name, value in DatasetPath.__dict__.items() if isinstance(value, type)])
    # print(ClassLabelLookuper(datasets="Cifar10"))
    # print(ClassLabelLookuper(datasets="Cifar100"))

    # for i in [name for name, value in DatasetPath.__dict__.items() if isinstance(value, type)]:
    #     print(ClassLabelLookuper(i)._cls2label)

    # import pickle
    #
    # with DatasetPath.Cifar100.train.open(mode="rb") as f:
    #     data = pickle.load(f, encoding="bytes")
    #     images, labels = data[b"data"], data[b"fine_labels"]
    #     images = images.reshape(-1, 3, 32, 32)
    # ccn = ClassLabelLookuper(datasets="Cifar100")
    # length = 64
    # visualize_pil(images[0: length + 1], [ccn.get_class(i) for i in labels[0: length + 1]]).show()
    # visualize_plt(images[0: length + 1], [ccn.get_class(i) for i in labels[0: length + 1]])
    # a = visualize_np(images[0: length + 1], [ccn.get_class(i) for i in labels[0: length + 1]])
    # print(a.shape)

    def rand_shape(): return (np.random.randint(100, 256), np.random.randint(100, 256))
    image = [np.random.randint(low=0, high=256, size=(*rand_shape(), 3), dtype=int) for i in range(64)]
    visualize_plt(image)
