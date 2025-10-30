import cv2
import numpy as np
import torch

from ..utils.onnx_loader import OnnxLoader
from .scrfd import SCRFD

scrfd_loader = OnnxLoader(
    model_name="scrfd-10g-bnkps-shape640x640-5838f7fe05.onnx",
    model_urls=[
        "http://nas.ddtl.work:8080/share.cgi?ssid=7caa3030065a49528c2ca75a55caae05&openfolder=forcedownload&ep=&_dc=1688959343160&fid=7caa3030065a49528c2ca75a55caae05",  # noqa: E501
    ],
    model_locals=[
        "scrfd-10g-bnkps-shape640x640-5838f7fe05.onnx",
    ],
)


class SCRFDDetector:
    ONNX_INPUT_SIZE = (640, 640)

    def __init__(self, device="cpu", model_path=None):
        model_weights = scrfd_loader.load_model_weights(model_path, device)
        self.detector = SCRFD(model_weights)
        if device == 'cuda':
            self.detector.prepare(torch.cuda.current_device())
        else:  # use cpu
            self.detector.prepare(-1)

    def predict(self, image: np.ndarray, threshold: float = 0.5, scale_size: float = 640, return_kps: bool = False) -> np.ndarray:
        """Detect the faces from a single image
           https://github.com/deepinsight/insightface/tree/master/detection/scrfd

        Args:
            image (np.ndarray): RGB uint8 input image
            threshold (float, optional): threshold determine it is face or not. Defaults to 0.5.

        Returns:
            np.ndarray: (N, 5) a list of detected face bounding box, [x1, y1, x2, y2, confidence].
        """
        if image.dtype != np.uint8:
            raise TypeError()
        # Scale to smaller size to get the better performance
        h, w = image.shape[:2]
        ratio = scale_size / max(h, w)
        if ratio < 1:
            resized = cv2.resize(image, (int(ratio * w), int(ratio * h)), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()
            ratio = 1
        bboxes, kpss = self.detector.detect(resized, threshold, self.ONNX_INPUT_SIZE)

        # Skip scale back when no face is found
        if len(bboxes) < 1:
            return bboxes

        # Scale back the bounding box
        bboxes[:, :4] = bboxes[:, :4] / ratio

        if return_kps:
            return bboxes, kpss / ratio
        return bboxes
