from pathlib import Path

from pydantic import BaseModel


class BaseSettings(BaseModel):
    data_dir: Path


class EurosatSettings(BaseSettings):
    data_dir = Path("../data/raw")
    valid_paths = Path("ds")


class StyleSettings(BaseSettings):
    data_dir = Path("../data/external")
    trainpath = Path("../data/external/sentences/train.feather")
    testpath = Path("../data/external/sentences/test.feather")
    log_dir = Path("../tune")
