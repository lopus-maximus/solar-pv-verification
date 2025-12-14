import cv2
import numpy as np

def draw_overlay(image, boxes, mask, out_path):
  
    overlay = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  
            2
        )

    if mask is not None:
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel

        overlay = cv2.addWeighted(
            overlay,
            1.0,
            colored_mask,
            0.4, 
            0
        )

    cv2.imwrite(out_path, overlay)
