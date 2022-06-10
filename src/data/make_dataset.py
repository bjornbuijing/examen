import random
from pathlib import Path
from typing import Tuple

import pandas as pd
import tensorflow as tf
from loguru import logger


def get_eurosat(data_dir: Path) -> Path:

    datapath = Path(data_dir) / "EuroSATallBands.zip"
    if datapath.exists():
        logger.info(f"{datapath} already exists, skipping download")
    else:
        logger.info(f"{datapath} not found on disk, downloading")

        url = "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip"

        datapath = tf.keras.utils.get_file(
            "EuroSATallBands.zip",
            url,
            extract=True,
            cache_dir=data_dir,
            cache_subdir="",
        )
    return datapath


def make_style(output_dir: Path, read_dir: Path) -> Tuple[Path, Path]:
    humor = pd.read_pickle(read_dir / "humorous_oneliners.pickle")
    reuters = pd.read_pickle(read_dir / "reuters_headlines.pickle")
    wiki = pd.read_pickle(read_dir / "wiki_sentences.pickle")
    proverbs = pd.read_pickle(read_dir / "proverbs.pickle")

    x0 = [(sent, "humor") for sent in humor]
    x1 = [(sent, "reuters") for sent in reuters]
    x2 = [(sent, "wiki") for sent in wiki]
    x3 = [(sent, "proverbs") for sent in proverbs]
    X = x0 + x1 + x2 + x3  # noqa: N806

    data = pd.DataFrame(X, columns=["sentence", "label"])
    data = data.sample(frac=1).reset_index(drop=True)

    splitidx = int(len(data) * 0.8)
    idx = [*range(len(data))]
    random.shuffle(idx)

    train = data.iloc[idx[:splitidx]].reset_index(drop=True)
    test = data.iloc[idx[splitidx:]].reset_index(drop=True)
    trainpath = output_dir / "train.feather"
    testpath = output_dir / "test.feather"
    train.to_feather(trainpath)
    test.to_feather(testpath)
    return trainpath, testpath
