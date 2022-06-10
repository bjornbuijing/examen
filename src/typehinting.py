from pathlib import Path
from typing import Callable, Dict, List, Protocol, Tuple

import numpy as np
import torch


class GenericModel(Protocol):
    train: Callable
    eval: Callable
    parameters: Callable

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass


class Dataset(Protocol):
    paths: List[Path]

    def __len__(self) -> int:
        pass


class ListDataset(Dataset):
    dataset: List

    def __getitem__(self, idx: int) -> Tuple:
        pass


class DictDataset(Dataset):
    name_mapping: Dict[str, int]
    dataset: Dict[int, np.ndarray]

    def __getitem__(self, idx: int) -> np.ndarray:
        pass
