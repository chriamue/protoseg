# source: https://www.jeremyjordan.me/evaluating-image-segmentation-models/
import numpy as np

def iou(prediction, label):
    intersection = np.logical_and(label, prediction)
    union = np.logical_or(label, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score