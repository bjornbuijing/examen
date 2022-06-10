import random
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from tqdm import tqdm

from src.typehinting import DictDataset, ListDataset


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

class BaseDataset(ListDataset):
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

class StyleDataset(BaseDataset):
    def process_data(self) -> None:
        path = self.paths[0]
        data = pd.read_feather(path, columns=["sentence", "label"])
        self.dataset = list(data.to_records(index=False))

class BaseDictDataset(DictDataset):
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset: Dict[int, np.ndarray] = {}
        self.name_mapping: Dict[str, int] = {}
        self.process_data()

    def __len__(self) -> int:
        raise NotImplementedError

    def process_data(self) -> None:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.dataset[idx]


class EuroSatDataset(BaseDictDataset):
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
        super().__init__(paths)

    def __len__(self) -> int:
        raise NotImplementedError

    def process_data(self) -> None:
        for path in tqdm(self.paths):
            class_name = path.parent.name
            if class_name not in self.name_mapping:
                self.name_mapping[class_name] = len(self.name_mapping)

            # add key-value pairs to self.dataset
            # the key is the class integer from name_mapping,
            # the value is the current List of Paths
            # if there is no value for the key, return an empty List
            key: int = None
            value: np.ndarray = None

            # we append to the array
            self.dataset[key] = np.append(value, np.array([path]))


class GenericStreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        self.batchsize = batchsize
        self.preprocessor = preprocessor

    def reset_index(self) -> None:
        raise NotImplementedError

    def batchloop(self) -> Sequence[Tuple]:
        raise NotImplementedError

    def stream(self) -> Iterator:
        raise NotImplementedError

class BaseDatastreamer(GenericStreamer):
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset: ListDataset,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        super().__init__(batchsize, preprocessor)
        self.dataset = dataset
        self.size = len(self.dataset)
        self.reset_index()

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

class SiameseStreamer(GenericStreamer):
    def __init__(
        self,
        dataset: DictDataset,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        super().__init__(batchsize, preprocessor)
        self.dataset = dataset
        self.size = len(self.dataset)
        self.reset_index()

        # the number of classes
        self.n: int = len(self.dataset.name_mapping)
        # precalculated combinations of all possible different classes combinations
        self.different = list(combinations([*range(self.n)], 2))

        assert batchsize % 2 == 0, "batchsize must be a multiple of 2"

    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0

    def random_index(
        self,
    ) -> Tuple[Tuple[List[Any], int], Tuple[List[Tuple[Any, Any]], Tuple[int, int]]]:
        """This function generates a batch with:
        50% of batchsize with indexes from the same class
        50% of batchsize with indexes from two different classes

        Returns
        - equal (int): this is a random class key from which to pick similar images
        - same (List[np.array]) : this is a list of indexes.
            You can use an index to get two images from the class 
        - i and j (int). Two different class keys 
        - other (List[Tuple]). A list of tuples, every tuple contains two indexes,
            one from class i, one from class j 
        """

        rng = np.random.default_rng()
        # random class
        equal = rng.integers(0, self.n, 1)[0]
        # random indexes from the random class
        index = rng.integers(0, len(self.dataset[equal]), self.batchsize)
        # splitted in pairs of two
        same = np.split(index, self.batchsize // 2)

        # pick two random different classes
        random.shuffle(self.different)
        i, j = self.different[0]

        # get random indexes from the first class
        index1 = rng.integers(0, len(self.dataset[i]), self.batchsize // 2)
        # and from the second class
        index2 = rng.integers(0, len(self.dataset[j]), self.batchsize // 2)
        # zip into tuples of two
        other = list(zip(index1, index2))
        return (same, equal), (other, (i, j))

    def batchloop(self) -> Sequence[Tuple]:
        """This generates batches for a siamese network.
        Every observation is a tuple (img1, img2, label)
        where label is a 1 if two images are equal, and
        a 0 otherwise.

        Returns:
            Sequence[Tuple]: _description_
        """
        batch = []
        (same, equal), (other, (i, j)) = self.random_index()

        # retrieve the arrays with paths from the three classes:
        #   - the equal class
        #   - the different classes i and j

        for idx in same:
            # append to the batch a tuple (img1, img2, 1)
            # use tifffile to read the image
            # cast the image to np.int32
            self.index += 1

        for idx in other:
            # append to the batch a tuple (img1, img2, 0)
            # use tifffile to read the image
            # cast the image to np.int32
            self.index += 1

        random.shuffle(batch)

        return batch

    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            if self.preprocessor is not None:
                X1, X2, Y = self.preprocessor(batch)  # noqa N806
            else:
                X1, X2, Y = zip(*batch)  # noqa N806
            yield X1, X2, Y
