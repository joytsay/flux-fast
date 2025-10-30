from pathlib import Path
from typing import List
from typing import Optional
from typing import Union


def mkdir(path: str) -> Path:
    """make directory and its parent folder, do nothing if the director is existed

    Args:
        path (str): the directory path

    Returns:
        Path: the Path object about the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rmdir(folder: str):
    """Recursively remove the directory

    Args:
        folder (str): the directory path
    """
    folder = Path(folder)
    for child in folder.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rmdir(folder)
    folder.rmdir()


def images_in_folder(folder: str, image_ext: Optional[Union[str, List[str]]] = None):
    """The sorted list of images in the folder

    Args:
        folder (str): directory path
        image_ext (Optional[Union[str, List[str]]], optional): valid image extension. Defaults to None.

    Returns:
        [type]: the sorted list of images
    """
    if image_ext is None:
        image_ext = [".jpg", ".png", ".tif"]

    return files_in_folder(folder, image_ext)


def files_in_folder(folder: str, file_ext: Union[str, List[str]]) -> List[str]:
    """The sorted list of files in the folder with given extension

    Args:
        folder (str): directory path
        file_ext (Union[str, List[str]]): file extension

    Returns:
        List[str]: the sorted list of files
    """

    if isinstance(file_ext, str):
        file_ext = [file_ext]

    files = [str(path) for path in Path(folder).iterdir() if path.suffix.lower() in file_ext]
    files = list(sorted(files))
    return files
