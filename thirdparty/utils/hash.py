import hashlib
import re


def get_md5_checksum(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html
def get_sha256_hash(path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb")as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# https://github.com/pytorch/pytorch/blob/1022443168b5fad55bbd03d087abf574c9d2e9df/torch/hub.py#L568
def parse_file_hash(file_name: str) -> str:
    """Parse the model path to get the hash string.

    Args:
        file_name (str): the file name with the same naming convention as torchvision models. i.e. name-sha256

    Returns:
        str: the sha256 hash or empty string if not found
    """
    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    r = HASH_REGEX.search(file_name)  # r is Optional[Match[str]]
    if r:
        return r.group(1)
    return ""


def parse_model_hash(file_name: str) -> str:
    return parse_file_hash(file_name)
