import io
from typing import Union
import pickle
import numpy as np
import torch

from ..utils.weight_loader import WeightLoader
from .archs.gfpganv1_clean_arch import GFPGANv1Clean

GFPGANv1_3_loader = WeightLoader(
    model_name="GFPGANv1.3-c953a88f.pth",
    model_urls=[
        "http://nas.ddtl.work:8080/share.cgi?ssid=e747c08c1204484caeadb3921318eceb&openfolder=forcedownload&ep=&_dc=1688960574909&fid=e747c08c1204484caeadb3921318eceb",  # noqa: E501
    ],
    model_locals=[
        "GFPGANv1.3-c953a88f.pth",
        "GFPGANv1.3.pth"
    ],
)

GFPGANv1_2_loader = WeightLoader(
    model_name="GFPGANCleanv1-NoCE-C2-29e25ee9.pth",
    model_urls=[
        "http://nas.ddtl.work:8080/share.cgi?ssid=3966bdd8431e4669aeac46dc5eba951f&openfolder=forcedownload&ep=&_dc=1688960609253&fid=3966bdd8431e4669aeac46dc5eba951f",  # noqa: E501
    ],
    model_locals=[
        "GFPGANCleanv1-NoCE-C2-29e25ee9.pth",
        "GFPGANCleanv1-NoCE-C2.pth"
    ],
)


# https://github.com/TencentARC/GFPGAN
class GFPGAN(object):
    @torch.compiler.set_stance("force_eager")
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "GFPGANv1.3",
        model_path: Union[str, io.BytesIO] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        if model_name in ["GFPGANv1.3", "GFPGANv1.2", "GFPGANCleanv1-NoCE-C2"]:
            self.input_size = 512
            self.model = GFPGANv1Clean(
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
            if model_name == "GFPGANv1.3":
                weight_loader = GFPGANv1_3_loader
            elif model_name in ["GFPGANv1.2", "GFPGANCleanv1-NoCE-C2"]:
                weight_loader = GFPGANv1_2_loader
        elif model_name == "GFPGANv1":
            raise NotImplementedError(model_name)
        elif model_name == "GFPGANvDD-1-512-sideface":
            raise NotImplementedError(model_name)
        else:
            raise NameError(f"Unknow model name: {model_name}")

        # fp32/bf16 full/half precision models
        try:
            self.model = self.model.to(dtype)
            if isinstance(model_path, io.BytesIO):
                weights = torch.load(model_path, weights_only=True)
            else:
                weights = weight_loader.load_model_weights(model_path, weights_only=True)
            if "params_ema" in weights:
                self.model.load_state_dict(weights["params_ema"])
            elif "params" in weights:
                self.model.load_state_dict(weights["params"])
            else:
                self.model.load_state_dict(weights)

        # if load pickle int8/int4 quantized models
        except pickle.UnpicklingError:
            setattr(torch.nn.Linear, "_linear_extra_repr", torch.nn.Linear.extra_repr)
            self.model = torch.load(model_path, weights_only=False)

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)

    @torch.no_grad()
    def process(self, image: np.ndarray) -> np.ndarray:
        if image.shape != (self.input_size, self.input_size, 3):
            raise RuntimeError(f"image size should be (512, 512, 3), found {image.shape}")

        image = (image - 127.5) / 127.5
        image = torch.Tensor(np.moveaxis(image, -1, 0)).unsqueeze(0).to(self.device)

        scaled = self.model(image, return_rgb=False)[0]
        scaled = torch.clamp(scaled * 0.5 + 0.5, 0, 1)
        scaled = np.moveaxis(scaled[0].cpu().numpy(), 0, -1)
        scaled = np.round(scaled * 255).astype(np.uint8)
        return scaled
