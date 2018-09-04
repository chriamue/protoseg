
import os
import numpy as np
import cv2
from . import backends


class DataLoader():

    images = []
    masks = []

    def __init__(self, root='data/', config=None, mode='train', augmentation=None):
        self.root = root
        self.config = config
        self.mode = mode
        self.augmentation = augmentation
        assert(config)

        _image_dir = os.path.join(root, mode)
        _masks_dir = os.path.join(root, mode + "_masks")

        self.images = (os.path.join(_image_dir, f)
                       for f in os.listdir(_image_dir) if "mask" not in f)
        if mode != 'test':
            self.masks = (os.path.join(_masks_dir, f)
                          for f in os.listdir(_masks_dir))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        if mode != 'test':
            assert (len(self.images) == len(self.masks))

    def resize(self, img, mask=None, width=None, height=None):
        img = cv2.resize(img, (width or self.config['width'], height or self.config['height']))
        if mask is None:
            return img
        mask = cv2.resize(
            mask, (width or self.config['width'], height or self.config['height']), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, index):

        if self.config['gray_img']:
            img = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        elif self.config['color_img']:
            img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED)

        if self.mode == 'test':
            img = self.resize(img)
            return backends.backend().dataloader_format(img), self.images[index]

        if self.config['gray_mask']:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        elif self.config['color_mask']:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_COLOR)
        else:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_UNCHANGED)

        if self.augmentation:
            img = self.augmentation.filter(img)
            img, mask = self.augmentation.random_flip(img, mask)
            img, mask = self.augmentation.random_rotation(img, mask)
            img, mask = self.augmentation.random_shift(img, mask)
            img, mask = self.augmentation.random_zoom(img, mask)
            img = self.augmentation.random_noise(img)
            img = self.augmentation.random_brightness(img)
        
        img, mask = self.resize(img, mask)

        return backends.backend().dataloader_format(img, mask)

    def __len__(self):
        return len(self.images)
