from pydantic import BaseModel
from pathlib import Path

class BaseSettings(BaseModel):
    data_dir: Path

class EurosatSettings(BaseSettings):
    data_dir = Path("../data/raw")
    valid_paths = Path("ds")
