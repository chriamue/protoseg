
import os
import numpy as np
import cv2
from . import backends

class DataLoader():

    images = []
    masks = []

    def __init__(self, root='data/', mode='train'):
        self.root = root
        self.mode = mode

        _image_dir = os.path.join(root, mode)
        _masks_dir = os.path.join(root, mode+"_masks")

        self.images = (os.path.join(_image_dir, f) for f in os.listdir(_image_dir))
        self.masks = (os.path.join(_masks_dir, f) for f in os.listdir(_masks_dir))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        if mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):

        img = cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

        return backends.backend().dataloader_format(img, mask)

    def __len__(self):
        return len(self.images)
