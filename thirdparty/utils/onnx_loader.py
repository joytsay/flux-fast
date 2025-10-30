from pathlib import Path
from typing import List

import onnx
import onnxruntime as ort
import torch

from .file_loader import FileLoader


class OnnxLoader(FileLoader):
    def __init__(self, model_name: str,  model_urls: List[str] = None, model_locals: List[str] = None) -> None:
        """Setup the model loader for specific pytorch model

        Args:
            model_name (str): the name of the model, it has the naming convention of torchvision, i.e. name-sha256
            model_urls (List[str], optional): the candidate urls for this the model
            model_locals (List[str], optional): the candidate local path for this the model
        """
        super().__init__(model_name, model_urls, model_locals)
        self.cache_sub_dir = "checkpoints"
        self.onnx_model = None
        self.onnx_runtime = None

    def load_model_weights(self, model_path: str = None, map_location="cpu"):
        """Load the weight of specific model parts

        Args:
            model_path (str, optional): the local model path. Defaults to None.
            map_location (str, optional): device the weight will be loaded in. Defaults to "cpu".

        Returns:
            Dict: the onnx model for graph info
            Dict: the onnxruntime session for inference use
        """
        model_path = self.get_file_path(model_path)
        self.onnx_model = onnx.load(model_path)
        providers = ['CPUExecutionProvider']
        if map_location is not None:
            device = torch.device(map_location)
            if device.type == "cpu":
                providers = ['CPUExecutionProvider']
            elif device.type == "cuda":
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': device.index if device.index is not None else 0
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                raise ValueError(f"Unrecognized device '{map_location}'")
        ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
        self.onnx_runtime = ort.InferenceSession(self.onnx_model.SerializeToString(), providers=providers)
        return self.onnx_runtime

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
