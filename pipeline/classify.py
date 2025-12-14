import math
from pipeline.radius_utils import compute_pixel_radius

def _intersects(box, w, h, r):
    x1, y1, x2, y2 = box
    cx, cy = w / 2, h / 2

    closest_x = min(max(cx, x1), x2)
    closest_y = min(max(cy, y1), y2)

    # Distance from center to closest box point
    dist = math.hypot(closest_x - cx, closest_y - cy)
    return dist <= r

def classify_solar(model, img, lat):
    res = model(img)[0]
    boxes = [b.xyxy[0].tolist() for b in res.boxes]
    confs = [float(b.conf[0]) for b in res.boxes]

    if not boxes:
        return False, 0.0, None, []

    h, w, _ = img.shape
    r1200 = compute_pixel_radius(lat, 1200)
    r2400 = compute_pixel_radius(lat, 2400)

    for b, c in zip(boxes, confs):
        if _intersects(b, w, h, r1200):
            return True, c, 1200, boxes

    for b, c in zip(boxes, confs):
        if _intersects(b, w, h, r2400):
            return True, c, 2400, boxes

    return False, 0.0, None, boxes
