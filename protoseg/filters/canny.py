import cv2

def canny(img, threshold1=100, threshold2=200):
    return cv2.Canny(img,threshold1=threshold1, threshold2=threshold2)

def addcanny(img, threshold1=100, threshold2=200):
    canny_ = canny(img, threshold1=threshold1, threshold2=threshold2)
    print(img.shape)
    if img.shape[2] == 3:
        img[:,:,0] = img[:,:,0] + canny_
        img[:,:,1] = img[:,:,1] + canny_
        img[:,:,2] = img[:,:,2] + canny_
    else:
        img = img + canny_
    return img