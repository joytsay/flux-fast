import os
from pathlib import Path
from typing import List

import requests
import torch
from torch.hub import download_url_to_file

from .hash import get_sha256_hash
from .hash import parse_file_hash
from .logging import logger as torchface_logger
from .os import mkdir


class FileLoader():
    def __init__(self, file_name: str,  file_urls: List[str] = None, file_locals: List[str] = None) -> None:
        """Setup the file loader for specific file

        Args:
            file_name (str): the name of the file, it has the naming convention of torchvision, i.e. name-sha256
            file_urls (List[str], optional): the candidate urls for this file
            file_locals (List[str], optional): the candidate local path for this file
        """
        self.file_name = file_name
        self.file_urls = () if file_urls is None else file_urls
        self.file_locals = () if file_locals is None else file_locals
        self.file_hash = parse_file_hash(file_name)
        self.cache_sub_dir = "files"

    def get_file_path(self, file_path: str = None) -> str:
        """Get the valid file path. Download file if not existed

        Args:
            file_path (str, optional): the local file path. Defaults to None.

        Returns:
            str: file_path
        """

        # 1. Use the provided path
        if file_path and Path(file_path).is_file():
            self.check_file_hash(file_path)
            torchface_logger.info(f"Use file from {file_path}")
            return file_path

        # 2. Use the search path provided by RESOURCES_FOLDER environmental variable
        if "RESOURCES_FOLDER" in os.environ:
            resource_folders = os.environ["RESOURCES_FOLDER"].split(os.pathsep)
            for resource_folder in resource_folders:
                for candidate_name in self.file_locals:
                    file_path = Path(resource_folder) / candidate_name
                    if self.check_file_valid(file_path):
                        return file_path

        if len(self.file_urls) == 0:
            torchface_logger.error(f"[{self.file_urls}]]: No file urls are provided")

        # 3. Search the Local cache
        cache_dir = Path(torch.hub.get_dir()) / self.cache_sub_dir
        for candidate_name in self.file_locals:
            file_path = cache_dir / candidate_name
            if self.check_file_valid(file_path):
                return file_path

        # 4. Download from the internet
        for candidate_url in self.file_urls:
            try:
                response = requests.head(candidate_url, timeout=10)
                file_path = cache_dir / self.file_name
                if response.status_code == 200:
                    torchface_logger.info(f"Downloading: {candidate_url} to {file_path}\n")
                    mkdir(cache_dir)
                    download_url_to_file(candidate_url, file_path, hash_prefix=self.file_hash)
                    if self.check_file_valid(file_path):
                        return file_path

            except requests.RequestException as e:
                torchface_logger.debug(e)
            except OSError as e:
                torchface_logger.debug(e)
            torchface_logger.info(f"[{self.file_name}]: failed to download from {candidate_url}")
        raise FileNotFoundError(f"[{self.file_name}], failed to load file")

    def check_file_valid(self, file_path):
        if file_path.is_file() and self.check_file_hash(file_path):
            torchface_logger.info(f"Use file from {file_path}")
            return True
        return False

    def check_file_hash(self, file_path):
        if len(self.file_hash) == 0:
            return True

        file_hash = get_sha256_hash(file_path)[:len(self.file_hash)]
        if file_hash == self.file_hash:
            return True
        torchface_logger.warning(f"Different hash value: get {file_hash}, expected {self.file_hash}")
        return False
