import math
from pipeline.radius_utils import compute_pixel_radius

def _inside(box, w, h, r):
    x1,y1,x2,y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    return math.hypot(cx-w/2, cy-h/2) <= r

def classify_solar(model, img, lat):
    res = model(img)[0]
    boxes = [b.xyxy[0].tolist() for b in res.boxes]
    confs = [float(b.conf[0]) for b in res.boxes]

    h,w,_ = img.shape
    r1200 = compute_pixel_radius(lat, 1200)
    r2400 = compute_pixel_radius(lat, 2400)

    for b,c in zip(boxes, confs):
        if _inside(b, w, h, r1200):
            return True, c, 1200, boxes

    for b in boxes:
        if _inside(b, w, h, r2400):
            return False, 0.0, 2400, boxes

    return False, 0.0, None, boxes
