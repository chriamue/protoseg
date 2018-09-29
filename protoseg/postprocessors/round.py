import numpy as np

def round(img):
    img = np.round(img, 0)
    return img.astype(np.int)