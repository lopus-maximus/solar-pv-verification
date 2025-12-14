import cv2
import numpy as np

def draw_overlay(image, boxes, mask, out_path):
    """
    image: original BGR image (cv2)
    boxes: list of YOLO boxes [x1, y1, x2, y2]
    mask: SAM mask (H x W) or None
    out_path: where to save overlay image
    """

    overlay = image.copy()

    # ---- Draw YOLO bounding boxes (RED) ----
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  # Red box
            2
        )

    # ---- Draw SAM mask (GREEN, transparent) ----
    if mask is not None:
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel

        overlay = cv2.addWeighted(
            overlay,
            1.0,
            colored_mask,
            0.4,  # transparency
            0
        )

    # ---- Save final image ----
    cv2.imwrite(out_path, overlay)
