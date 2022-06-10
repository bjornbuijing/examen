import random
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Dict

import numpy as np


def walk_dir(path: Path) -> Iterator:
    """loops recursively through a folder

    Args:
        path (Path): folder to loop trough. If a directory
            is encountered, loop through that recursively.

    Yields:
        Generator: all paths in a folder and subdirs.
    """

    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk_dir(p)
            continue
        # resolve works like .absolute(), but it removes the "../.." parts
        # of the location, so it is cleaner
        yield p.resolve()


def iter_valid_paths(path: Path, formats: List[str]) -> Tuple[Iterator, List[str]]:
    """
    Gets all paths in folders and subfolders
    strips the classnames assuming that the subfolders are the classnames
    Keeps only paths with the right suffix


    Args:
        path (Path): image folder
        formats (List[str]): suffices to keep.

    Returns:
        Tuple[Iterator, List[str]]: _description_
    """
    # gets all files in folder and subfolders
    walk = walk_dir(path)
    # retrieves foldernames as classnames
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    # keeps only specified formats
    paths = (path for path in walk if path.suffix in formats)
    return paths, class_names


class BaseDataset:
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset: List = []
        self.process_data()

    def process_data(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        return self.dataset[idx]

class EuroSatDataset(BaseDataset):
    def __init__(self, paths: List[Path]) -> None:
        """
        This dataset is stored as a dictionary instead of a list.
        The dictionary stores int : List[Path] key-value pairs
        the key is a class, the value is a list of paths of an identical class
        eg:

        dataset = {
            0 : ["path/to/a/01.jpg", "path/to/a/02.jpg"],
            1 : ["path/to/b/03.jpg", "path/to/b/04.jpg"]
        }

        Args:
            paths (List[Path]): filepaths, where the class name is the parent folder
        """
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset: Dict[int, np.ndarray] = {}
        self.name_mapping : Dict[str, int] = {}
        self.process_data()

    def __len__(self) -> int:
        return NotImplementedError

    def process_data(self) -> None:
        for path in tqdm(self.paths):
            class_name = path.parent.name
            if class_name not in self.name_mapping:
                self.name_mapping[class_name] = len(self.name_mapping)

            # add key-value pairs to self.dataset
            # the key is the class integer from name_mapping,
            # the value is the current List of Paths 
            # if there is no value for the key, return an empty List
            key = None
            value = None

            # we append to the list
            self.dataset[key] = value + [path]
        
        # and cast everything to a numpy array at the end
        for key in self.dataset:
            self.dataset[key] = np.array(self.dataset[key])

class BaseDatastreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset: BaseDataset,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batchsize = batchsize
        self.preprocessor = preprocessor
        self.size = len(self.dataset)
        self.reset_index()

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0

    def batchloop(self) -> Sequence[Tuple]:
        batch = []
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            batch.append((x, y))
            self.index += 1
        return batch

    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            if self.preprocessor is not None:
                X, Y = self.preprocessor(batch)  # noqa N806
            else:
                X, Y = zip(*batch)  # noqa N806
            yield X, Y
