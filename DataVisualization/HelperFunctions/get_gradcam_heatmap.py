"""
Merge a saliency map and an image into one image.
"""

import cv2
import numpy as np


def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:

    numer = mask - np.min(mask)
    denom = (mask.max() - mask.min()) + 10e-15
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    img = (img * 255).astype("uint8")

    colormap = cv2.COLORMAP_JET

    alpha = 0.4

    heatmap = cv2.applyColorMap(255 - heatmap, colormap)

    if heatmap.shape[0] == 3:
        heatmap = np.transpose(heatmap, axes=(1, 2, 0))
    if img.shape[0] == 3:
        img = np.transpose(img, axes=(1, 2, 0))

    output = ((1 - alpha) * img + alpha * heatmap).astype("uint8")

    return output
