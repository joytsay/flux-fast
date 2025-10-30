import argparse
import os
import random
import time
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with an exported GFPGAN AOTI (.pt2) package."
    )
    parser.add_argument(
        "--package",
        type=str,
        default="gfpgan_aoti.pt2",
        help="Path to the exported GFPGAN package (*.pt2).",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the aligned 512x512 input face image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gfpgan_aoti_output.png",
        help="Path to the enhanced output image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Number of warmup iterations before measuring time.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=5,
        help="Number of timed iterations used for mean/variance reporting.",
    )
    return parser


def _set_rand_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_input(image_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image {image_path}")
    if image.shape[:2] != (512, 512):
        raise ValueError(f"Expected 512x512 input image, received {image.shape[:2]}")

    image_f32 = image.astype("float32") / 255.0
    tensor = torch.from_numpy(image_f32.transpose(2, 0, 1))
    normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")

    tensor = tensor.detach().to(torch.float32).cpu()
    tensor = tensor.clamp_(-1.0, 1.0)
    tensor = (tensor + 1.0) * 0.5
    tensor = tensor.permute(1, 2, 0).contiguous().numpy()
    tensor = np.clip(tensor, 0.0, 1.0)
    tensor = np.clip(tensor * 255.0 + 0.5, 0.0, 255.0).astype("uint8")
    return tensor


def _load_package(path: str) -> torch.nn.Module:
    from torch._inductor.package import load_package

    if not os.path.exists(path):
        raise FileNotFoundError(f"Package file not found at {path}")
    return load_package(path, run_single_threaded=True)


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _extract_output(outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], list]) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise RuntimeError("Unexpected output type from GFPGAN package inference.")


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()

    _set_rand_seeds(args.seed)

    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    compiled_module = _load_package(args.package)

    input_tensor = _prepare_input(args.image, device, dtype)

    timings = []
    restored = None
    with torch.no_grad():
        for _ in range(max(args.warmup_iters, 0)):
            outputs = compiled_module(input_tensor)
            restored = _extract_output(outputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(max(args.benchmark_iters, 0)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = compiled_module(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            restored = _extract_output(outputs)

    if restored is None:
        with torch.no_grad():
            outputs = compiled_module(input_tensor)
            restored = _extract_output(outputs)
        print("Executed a single inference to produce output image (no timing collected).")

    if timings:
        tensor = torch.tensor(timings, dtype=torch.float64)
        mean = tensor.mean().item()
        var = tensor.var(unbiased=False).item() if tensor.numel() > 1 else 0.0
        print(
            f"Inference timings over {len(timings)} run(s): "
            f"mean={mean * 1e3:.3f} ms, var={var} s"
        )
    else:
        print("No benchmark iterations executed; skipping timing statistics.")

    image = _tensor_to_image(restored)
    _ensure_output_dir(args.output)
    args.image  = args.image.replace(".png", "_")
    args.output = args.image + args.output
    cv2.imwrite(args.output, image)
    print(f"Saved enhanced image to {args.output}")


if __name__ == "__main__":
    main()
