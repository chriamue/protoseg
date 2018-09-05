# source: https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
import numpy as np

def dice(seg, gt):
    k = 0
    return np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
