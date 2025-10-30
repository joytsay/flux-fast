from pathlib import Path
from typing import Dict
from typing import List

import torch

from .file_loader import FileLoader


class WeightLoader(FileLoader):
    def __init__(self, model_name: str,  model_urls: List[str] = None, model_locals: List[str] = None) -> None:
        """Setup the model loader for specific pytorch model

        Args:
            model_name (str): the name of the model, it has the naming convention of torchvision, i.e. name-sha256
            model_urls (List[str], optional): the candidate urls for this the model
            model_locals (List[str], optional): the candidate local path for this the model
        """
        super().__init__(model_name, model_urls, model_locals)
        self.cache_sub_dir = "checkpoints"

    def load_model_weights(self, model_path: str = None, map_location="cpu", weights_only: bool = None) -> Dict:
        """Load the weight of specific model parts

        Args:
            model_path (str, optional): the local model path. Defaults to None.
            map_location (str, optional): device the weight will be loaded in. Defaults to "cpu".

        Returns:
            Dict: the state_dict of the specific model parts
        """
        model_path = self.get_file_path(model_path)
        return torch.load(model_path, map_location=map_location, weights_only=weights_only)

    @property
    def model_name(self):
        return self.file_name

    @property
    def model_urls(self):
        return self.file_urls

    @property
    def model_locals(self):
        return self.file_locals

    @property
    def model_hash(self):
        return self.file_hash
