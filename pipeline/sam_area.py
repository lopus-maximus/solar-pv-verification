import math
import numpy as np

def compute_area(mask, lat, zoom=20):
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)
    return float(mask.sum() * (mpp**2))
