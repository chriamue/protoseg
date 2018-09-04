import cv2


def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_8U)
