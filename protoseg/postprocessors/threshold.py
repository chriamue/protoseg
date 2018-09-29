import numpy as np

def threshold(img, threshold=0.5):
    img[img <= threshold] = 0
    return img