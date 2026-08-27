"""Manages file system paths and local data for qgtoolbox"""

import shutil
import yaml
import xarray as xr
from pathlib import Path
from typing import Optional, Union

def parse_yaml(path):
    with open(Path(path)) as f:
        return yaml.safe_load(f)

def save_dataset(data, path):
    data.to_netcdf(path, engine="h5netcdf")

def load_dataset(path):
    with xr.open_dataset(path, engine="h5netcdf") as data:
        return data.load()

class DataManager:
    """Manages file system paths and local data for ptatoolbox."""
    def __init__(self, root_dir: Optional[Union[str, Path]] = "."):
        """Initialize the DataManager."""
        self.root = Path(root_dir or ".").expanduser().resolve()
        self.data = self.root / "data"
        self.config = self.root / "config"
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        # self.storage = self.processed / ".storage"

        self.root.mkdir(exist_ok=True)
        self.data.mkdir(exist_ok=True)
        self.config.mkdir(exist_ok=True)
        self.raw.mkdir(exist_ok=True)
        self.processed.mkdir(exist_ok=True)
        # self.storage.mkdir(exist_ok=True)

        # self._copy_raw_to_storage()

    def make_experiment(self, name: str) -> Path:
        """Create a new experiment directory inside the root."""
        experiment = self.processed / name
        experiment.mkdir(exist_ok=True)
        return experiment

    def make_path(self, path: str) -> Path:
        path = Path(path).expanduser().resolve()
        return path

    def storage_file_path(self, filename: str = "") -> Path:
        """Return a path to a file inside the storage directory."""
        return self.storage / filename

    def _copy_raw_to_storage(self) -> None:
        """Copy all regular files from locals directory to storage."""
        for path in self.raw.iterdir():
            if path.is_file():
                self._copy_from_raw(path.name)

    def _copy_from_raw(self, filename: str) -> Path:
        """Copy a single file from locals to storage."""
        src = self.raw / filename
        dst = self.storage / filename
        shutil.copy(src, dst)