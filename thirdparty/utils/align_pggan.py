import cv2
import numpy as np


def get_pggan_alignment_from_template(landmarks: np.ndarray, output_size: int = 256) -> np.ndarray:
    template_lms = LANDMARKS_TEMPLATE * output_size
    m = np.eye(3)
    m[:2] = cv2.estimateAffinePartial2D(landmarks, template_lms, method=cv2.LMEDS)[0]
    return m

# https://github.com/xinntao/facexlib/blob/29d792eb2a48e4e87b2b57001a85d2297c439a6f/facexlib/utils/face_restoration_helper.py#L73
LANDMARKS_TEMPLATE = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
]) / 512.
