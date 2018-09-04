import cv2


def sobel(img, dx, dy, ksize = 5):
    s = cv2.Sobel(img, cv2.CV_8U, dx, dy, ksize=ksize)
    return s
