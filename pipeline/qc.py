import numpy as np

def qc_check(img):
    return "VERIFIABLE" if np.mean(img) > 40 else "NOT_VERIFIABLE"
