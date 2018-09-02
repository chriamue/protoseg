# source: https://github.com/Naurislv/P12.1-Semantic-Segmentation/blob/master/augmentation.py
from random import randint, randrange, uniform
import numpy as np
import cv2


class Augmentation():

    def __init__(self, config):
        self.config = config
        assert(config)

    def resize(self, img, mask=None):
        img = cv2.resize(img, (self.config['width'], self.config['height']))
        if mask is None:
            return img
        mask = cv2.resize(
            mask, (self.config['width'], self.config['height']), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def random_flip(self, img, mask=None):
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
        h_img = img.shape[0]
        w_img = img.shape[1]
        if img.ndim > 2:
            ch_img = img.shape[2]
        center = (w_img / 2, h_img / 2)

        rotation = uniform(-degree, degree)
        rot_mtrx = cv2.getRotationMatrix2D(center, rotation, 1.0)

        img = cv2.warpAffine(img, rot_mtrx, (w_img, h_img))
        if img.ndim > 2:
            img = img.reshape(h_img, w_img, ch_img)
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

        if noise_chance < 0.01 or amount < 1:
            return img

        if uniform(0, 1) > noise_chance:
            # needs preallocated input image
            noise = np.zeros_like(img, img.dtype)
            noise = cv2.randn(noise, (0), (amount))
            img = np.where((255 - img) < noise, 255, img + noise)

        return img

    def random_brightness(self, img):
        """Add random brightness to give image dataset to imitate day/night."""

        min_bright = self.config['min_bright']
        max_bright = self.config['max_bright']
        min_val = self.config['min_val']
        max_val = self.config['max_val']
        # random_bright = np.random.uniform(min_bright, max_bright, 1)[0]
        random_bright = randrange(min_bright, max_bright)
        data_type = img.dtype
        if random_bright > 0:
            # add brightness
            img = np.where((max_val - img) < random_bright,
                           max_val, img + random_bright)
        elif random_bright < 0:
            # remove brightness
            img = np.where((img + random_bright) <= min_val,
                           min_val, img + random_bright)

        return img.astype(data_type)

    def random_padding(self, img, output_size, override_random=None):
        """Add random horizontal/vertical shifts and increases size of image to output_size."""

        h_img = img.shape[0]
        w_img = img.shape[1]
        if img.ndim > 2:
            ch_img = img.shape[2]
        h_output, w_output = output_size

        asser_msg = ("For Random padding input image Hight must be less or equal to "
                     "output_size hight")
        assert h_img <= h_output, asser_msg
        assert_msg = ("For Random padding input image Width must be less or equal to "
                      "output_size width")
        assert w_img <= w_output, assert_msg

        if img.ndim > 2:
            output_image = np.zeros(
                (h_output, w_output, ch_img), dtype=np.float32)
        else:
            output_image = np.zeros((h_output, w_output), dtype=np.float32)

        if override_random is None:
            pad_h_up = randint(0, h_output - h_img)
            pad_w_left = randint(0, w_output - w_img)
            pad_h_down = h_output - h_img - pad_h_up
            pad_w_right = w_output - w_img - pad_w_left
        else:
            pad_h_up = override_random[0]
            pad_w_left = override_random[1]
            pad_h_down = h_output - h_img - pad_h_up
            pad_w_right = w_output - w_img - pad_w_left

        output_image = np.pad(img, ((pad_h_up, pad_h_down), (pad_w_left, pad_w_right), (0, 0)),
                              'constant', constant_values=0)

        return output_image, (pad_h_up, pad_w_left)

    def random_zoom(self, img, mask):
        """Randomly zoom image."""

        if not img.ndim == 3 or not mask.ndim == 3:
            return img, mask

        zoom_in = self.config['zoom_in']
        zoom_out = self.config['zoom_out']

        output_size_h = img.shape[0]
        output_size_w = img.shape[1]

        rand_size = uniform(-1, 1)

        if rand_size < 0:
            max_zoom = output_size_h * zoom_out
            random_size_h = int(output_size_h + max_zoom * rand_size)
        else:
            max_zoom = output_size_h * zoom_in
            random_size_h = int(output_size_h + max_zoom * rand_size)

        random_size_w = output_size_w * random_size_h // output_size_h
        if random_size_w == output_size_w:
            return img, mask

        # Image zooming
        img = cv2.resize(img, (random_size_w, random_size_h),
                         interpolation=cv2.INTER_AREA)

        if random_size_w < output_size_w:
            img, _ = self.random_padding(
                img, output_size=(output_size_h, output_size_w))
        elif random_size_w > output_size_w:
            diff_w = random_size_w - output_size_w
            diff_h = random_size_h - output_size_h

            img = img[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]
        else:
            print("Failed random_zooms ? %s", img.shape)

        # Mask zooming
        mask = cv2.resize(mask, (random_size_w, random_size_h),
                          interpolation=cv2.INTER_AREA)

        if random_size_w < output_size_w:
            mask, _ = self.random_padding(
                mask, output_size=(output_size_h, output_size_w))
        elif random_size_w > output_size_w:
            diff_w = random_size_w - output_size_w
            diff_h = random_size_h - output_size_h

            mask = mask[diff_h // 2: -diff_h //
                        2, diff_w // 2: -diff_w // 2, :]
        else:
            print("Failed random_zooms ? %s", mask.shape)

        return img, mask
