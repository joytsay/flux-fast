import os
import random
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision.transforms.functional import normalize

from utils.benchmark_utils import annotate, create_parser

try:
    from thirdparty.gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean  # type: ignore
except ImportError as exc:
    raise ImportError(
        "GFPGAN dependencies are missing. Make sure thirdparty/gfpgan is available on PYTHONPATH."
    ) from exc


def _create_parser():
    parser = create_parser()
    parser.description = "Benchmark GFPGAN enhanced inference exported via AOTI or torch.compile."
    parser.set_defaults(
        image="quant/source.png",
        output_file="gfpgan_output.png",
        compile_export_mode="export_aoti",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="GFPGANv1.3",
        choices=["GFPGANv1.3", "GFPGANv1.2", "GFPGANCleanv1-NoCE-C2"],
        help="Name of the GFPGAN model to load.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="quant/GFPGANv1.3-c953a88f.pth",
        help="Path to checkpoint containing params or params_ema.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=10,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for benchmarking (reuses the same image for each sample).",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default=None,
        help="Optional override for the exported package filename.",
    )
    return parser


def _load_checkpoint(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    if "params_ema" in checkpoint:
        return checkpoint["params_ema"]
    if "params" in checkpoint:
        return checkpoint["params"]
    return checkpoint


def _build_model(model_name: str, model_path: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    if model_name not in {"GFPGANv1.3", "GFPGANv1.2", "GFPGANCleanv1-NoCE-C2"}:
        raise ValueError(f"Unsupported model_name={model_name}")

    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    state_dict = _load_checkpoint(model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()
    model.requires_grad_(False)
    model.to(dtype=dtype, device=device)
    return model


def _prepare_input(
    image_path: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image {image_path}")
    if image.shape[:2] != (512, 512):
        raise ValueError(f"Expected 512x512 input image, received {image.shape[:2]}")

    image_f32 = image.astype("float32") / 255.0
    tensor = torch.from_numpy(image_f32.transpose(2, 0, 1))
    normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    tensor = tensor.unsqueeze(0)
    if batch_size > 1:
        tensor = tensor.repeat(batch_size, 1, 1, 1)
    tensor = tensor.to(device=device)
    return tensor.to(dtype=dtype)


def _apply_compile(model: torch.nn.Module) -> torch.nn.Module:
    sample_param = next(model.parameters(), None)
    device_type = sample_param.device.type if sample_param is not None else "cpu"
    if device_type != "cuda":
        # torch.compile offers limited benefits on CPU; fall back to eager.
        return model
    return torch.compile(model, mode="max-autotune")


def _export_aoti(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    cache_dir: str,
    package_name: Optional[str],
    reuse: bool,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, ...]]:
    from torch._inductor.package import load_package

    os.makedirs(cache_dir, exist_ok=True)
    package_path = os.path.join(cache_dir, package_name or "gfpgan_aoti.pt2")

    if not (reuse and os.path.exists(package_path)):
        exported = torch.export.export(model, (example_input,), {})
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=package_path,
            inductor_configs={"max_autotune": True, "triton.cudagraphs": True},
        )

    compiled = load_package(package_path, run_single_threaded=True)

    def forward(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = compiled(tensor)
        if isinstance(outputs, torch.Tensor):
            return (outputs,)
        if isinstance(outputs, (tuple, list)):
            return tuple(outputs)
        raise RuntimeError("Unexpected output type from compiled package")

    return forward


def _set_rand_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _benchmark(
    runner: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]],
    example_input: torch.Tensor,
    warmup_iters: int,
    benchmark_iters: int,
) -> Tuple[float, float, list]:
    timings = []
    with torch.no_grad():
        for _ in range(warmup_iters):
            outputs = runner(example_input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if example_input.device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(benchmark_iters):
            if example_input.device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.time()
            outputs = runner(example_input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if example_input.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            timings.append(end - begin)

    tensor = torch.tensor(timings, device="cpu")
    mean = tensor.mean().item()
    var = tensor.var(unbiased=False).item() if tensor.numel() > 1 else 0.0
    return mean, var, timings


def _save_output(output: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    if output.dim() == 4:
        batch_size = output.size(0)
        base, ext = os.path.splitext(path)
        for idx in range(batch_size):
            image = _tensor_to_image(output[idx])
            suffix = "" if idx == 0 else f"_{idx}"
            cv2.imwrite(f"{base}{suffix}{ext}", image)
        return

    image = _tensor_to_image(output)
    cv2.imwrite(path, image)


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


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()

    _set_rand_seeds(args.seed)
    device_str = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    if args.disable_bf16 or device.type != "cuda":
        dtype = torch.float32

    model = _build_model(args.model_name, args.model_path, device, dtype)

    # apply float8 quantization
    if not args.disable_quant:
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, float8_dynamic_activation_float8_weight #, PerRow

        quantize_(
            model,
            int8_dynamic_activation_int8_weight(set_inductor_config=True),
            # float8_dynamic_activation_float8_weight(),
            # float8_dynamic_activation_float8_weight(granularity=PerRow()),
        )

    image_path = args.image or "quant/source.png"
    input_tensor = _prepare_input(image_path, device, dtype, args.batch_size)

    if args.compile_export_mode == "compile":
        model = _apply_compile(model)

    if args.compile_export_mode == "export_aoti":
        compiled_forward = _export_aoti(
            model,
            input_tensor,
            cache_dir=args.cache_dir,
            package_name=args.package_name,
            reuse=args.use_cached_model,
        )

        def forward(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            return compiled_forward(tensor)

    else:

        def forward(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            outputs = model(tensor)
            if isinstance(outputs, torch.Tensor):
                return (outputs,)
            if isinstance(outputs, (tuple, list)):
                return tuple(outputs)
            raise RuntimeError("Unexpected output type from model")

    mean_time, var_time, timings = _benchmark(
        forward,
        input_tensor,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    print(f"time mean/var: {timings} {mean_time} {var_time}")

    with torch.no_grad():
        result = forward(input_tensor)
    trace_file = args.trace_file.replace(".json.gz", "_")
    output_file = trace_file + args.output_file
    _save_output(result, output_file)

    if args.trace_file is not None:
        traced_forward = annotate(model.forward, "gfpgan_forward")
        model.forward = traced_forward
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("timed_region"):
                with torch.no_grad():
                    model(input_tensor)
        prof.export_chrome_trace(args.trace_file)


if __name__ == "__main__":
    main()
