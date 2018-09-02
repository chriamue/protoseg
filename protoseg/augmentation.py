# source: https://github.com/Naurislv/P12.1-Semantic-Segmentation/blob/master/augmentation.py
from random import randint, randrange, uniform
import numpy as np
import cv2

class Augmentation():

    def __init__(self, config):
        self.config = config
        assert(config)

    def resize(self, img, mask = None):
        img = cv2.resize(img, (self.config['width'], self.config['height']))
        if mask is None:
            return img
        mask = cv2.resize(mask, (self.config['width'], self.config['height']),interpolation=cv2.INTER_NEAREST)
        return img, mask

    def random_flip(self, img, mask = None):
        """Apply random flip to single image and label."""
        if not self.config['flip']:
            if mask is None:
                return img
            return img, mask

        horizontal = self.config['horizontal_flip']

        flip = 1
        rand_float = uniform(0, 1)

        if horizontal:
            # 1 == vertical flip
            # 0 == horizontal flip
            flip = randint(0, 1)

        if rand_float > 0.5:
            img = cv2.flip(img, flip)
            mask = cv2.flip(mask, flip)

        return img, mask

    def random_rotation(self, img, mask=None):
        """Rotate image randomly."""
        degree = self.config['rotation_degree']
        #(h_img, w_img, ch_img) = img.shape[:3]
        (h_img, w_img) = img.shape[:img.ndim]
        center = (w_img / 2, h_img / 2)

        rotation = uniform(-degree, degree)
        rot_mtrx = cv2.getRotationMatrix2D(center, rotation, 1.0)

        img = cv2.warpAffine(img, rot_mtrx, (w_img, h_img))#.reshape(h_img, w_img, ch_img)
        if mask is not None:
            mask = cv2.warpAffine(mask, rot_mtrx, (w_img, h_img))

            return img, mask

        return img

    def random_shift(self, img, mask=None):
        """Add random horizontal/vertical shifts to image dataset to imitate
        steering away from sides."""
        h_shift = self.config['horizontal_shift']
        v_shift = self.config['vertical_shift']

        rows = img.shape[0]
        cols = img.shape[1]

        horizontal = uniform(- h_shift / 2, h_shift / 2)
        vertical = uniform(- v_shift / 2, v_shift / 2)

        mtx = np.float32([[1, 0, horizontal], [0, 1, vertical]])

        # change also corresponding lable -> steering angle
        img = cv2.warpAffine(img, mtx, (cols, rows))
        if mask is None:
            return img
        mask = cv2.warpAffine(mask, mtx, (cols, rows))

        return img, mask

    def random_noise(self, img):
        """Add random noise to image dataset.
        noise_chance: probability that noise will be applied.
        """
        amount = self.config['noise_amount']
        noise_chance = self.config['noise_chance']
        if noise_chance < 0.01:
            return img

        if uniform(0, 1) > noise_chance:
            noise = np.zeros_like(img, img.dtype)  # needs preallocated input image
            noise = cv2.randn(noise, (0), (amount))

            img = np.where((255 - img) < noise, 255, img + noise)

        return img