import math

def meters_per_pixel(lat, zoom=20):
    return 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)

def area_sqft_to_radius_m(area):
    return math.sqrt(area / math.pi) * 0.3048

def compute_pixel_radius(lat, area_sqft, zoom=20):
    return area_sqft_to_radius_m(area_sqft) / meters_per_pixel(lat, zoom)
