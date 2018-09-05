import numpy as np
import cv2


def dilation(img, kernelw=5, kernelh=5, iterations=1):
    kernel = np.ones((kernelw, kernelh), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=iterations)
    return dilation


def erosion(img, kernelw=5, kernelh=5, iterations=1):
    kernel = np.ones((kernelw, kernelh), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=iterations)
    return erosion


def opening(img, kernelw=5, kernelh=5, iterations=1):
    kernel = np.ones((kernelw, kernelh), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening