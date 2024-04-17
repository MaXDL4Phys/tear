import torchvision
from random import random, randint, choice, uniform
from PIL import Image, ImageOps, ImageFilter

import numpy as np
import numbers
import math
import torch
class GroupRandomCrop(object):
    """
    The GroupRandomCrop class is used to perform random cropping on a group of images.

    Parameters:
    - size (int or tuple): The output size of the cropped images. If size is an int, the output size will be (size, size). If size is a tuple, the output size will be size.

    Methods:
    - __init__(self, size): Initializes a new instance of the GroupRandomCrop class with the specified size.
    - __call__(self, img_group): Applies random cropping to the input group of images.

    Example Usage:
    crop_transform = GroupRandomCrop(224)
    cropped_images = crop_transform(image_group)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()
        x1 = randint(0, w - tw)
        y1 = randint(0, h - th)

        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    """
    GroupCenterCrop

    Class for center cropping a group of images.

    Parameters:
        size (int or tuple) -- The size of the output crop. If size is an int, a square crop of size (size, size) is returned. If size is a tuple, it should be in the format (height, width
    *).

    Methods:
        __init__(self, size)
            Initialize the GroupCenterCrop transform.

        __call__(self, img_group)
            Apply the center crop transformation to a group of images.

    Examples:
        >>> transform = GroupCenterCrop(256)
        >>> img_group = [img1, img2, img3]
        >>> cropped_group = transform(img_group)
    """
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """
    GroupRandomHorizontalFlip

    Class for applying random horizontal flip to a group of images.

    Attributes:
        is_sth (bool): Flag to determine whether to apply horizontal flip or not.

    Methods:
        __init__(self, is_sth=False)
            Initializes the GroupRandomHorizontalFlip object.

        __call__(self, img_group, is_sth=False)
            Applies the random horizontal flip to the input image group.

    Example usage:

        # Initialize GroupRandomHorizontalFlip with flip enabled
        flip = GroupRandomHorizontalFlip(is_sth=True)

        # Apply flip to image group
        flipped_group = flip(img_group)

    """

    def __init__(self, is_sth=False):
        self.is_sth = is_sth

    def __call__(self, img_group, is_sth=False):
        v = random()
        if not self.is_sth and v < 0.5:

            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize1(object):
    """
    GroupNormalize1

    Class used to normalize a group of images using the torchvision.transforms.Normalize function.

    Attributes:
        mean (list): The mean values for each channel of the images.
        std (list): The standard deviation values for each channel of the images.
        worker (torchvision.transforms.Normalize): The Normalize transform object.

    Methods:
        __init__(mean, std):
            Initializes the GroupNormalize1 object.

        __call__(img_group):
            Applies the normalization transform to a group of images.

    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.worker = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupNormalize(object):
    """A class for grouping normalization of tensors."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0] // len(self.mean))
        std = self.std * (tensor.size()[0] // len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if len(tensor.size()) == 3:
            # for 3-D tensor (T*C, H, W)
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            # for 4-D tensor (C, T, H, W)
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor


class GroupScale(object):
    """
    GroupScale
    ==========

    This class is used to resize a group of images using the provided size and interpolation method.

    Methods:
    ---------
    __init__(size, interpolation=Image.BICUBIC)
        Initializes a new instance of the GroupScale class.

        Parameters:
        - size (tuple or int): The desired output size. If size is an int, the output size will be (size, size).
                              If size is a tuple, it should contain two values (width, height).
        - interpolation (int, optional): The interpolation method to be used. Default is Image.BICUBIC.

    __call__(img_group)
        Resizes a group of images using the specified size and interpolation method.

        Parameters:
        - img_group (list): The list of images to be resized.

        Returns:
        - list: A new list of resized images.
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    """

    """
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h
        )
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == "L" and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupFCSample(object):
    """
    This class represents a data transformation for processing image groups.

    :param crop_size: The size of the cropped images. Can be an integer to specify a square crop, or a tuple of two integers to specify a rectangular crop.
    :param scale_size: The size to scale the images to before cropping. If None, no scaling will be performed.

    Example usage:
        crop_size = (224, 224)
        scale_size = (256, 256)
        transform = GroupFCSample(crop_size, scale_size)
        transformed_images = transform(image_group)
    """
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fc_fix_offset(
            image_w, image_h, image_h, image_h
        )
        oversample_group = list()

        for o_w, o_h in offsets:
            normal_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + image_h, o_h + image_h))
                normal_group.append(crop)
            oversample_group.extend(normal_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    """

    """
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = choice(pairs)
        if not self.fix_crop:
            w_offset = randint(0, image_w - crop_pair[0])
            h_offset = randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    @staticmethod
    def fill_fc_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 2
        h_step = (image_h - crop_h) // 2

        ret = list()
        ret.append((0, 0))  # left
        ret.append((1 * w_step, 1 * h_step))  # center
        ret.append((2 * w_step, 2 * h_step))  # right

        return ret


class GroupRandomSizedCrop(object):
    """
    :class:`GroupRandomSizedCrop` randomly crops and resizes a group of images.

    :param size: The size of the output crop (single integer for both width and height).
    :type size: int
    :param interpolation: The interpolation method to use for resizing the images. Default is `Image.BILINEAR`.
    :type interpolation: int, optional

    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = uniform(0.08, 1.0) * area
            aspect_ratio = uniform(3.0 / 4, 4.0 / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = randint(0, img_group[0].size[0] - w)
                y1 = randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))

class Stack(object):
    """

    The Stack class represents a utility for stacking multiple images together. It concatenates the images along the specified axis.

    Attributes:
        roll (bool): Flag to determine whether to vertically roll the RGB images before stacking.

    Methods:
        __init__(self, roll=False)
            Initializes the Stack object with the specified roll flag.

        __call__(self, img_group)
            Concatenates the images in the img_group along the specified axis.
            Returns the stacked image.

    Example Usage:
        stack = Stack(roll=True)
        stacked_image = stack(image_group)
    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == "L":
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == "RGB":
            if self.roll:
                return np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                )
            else:
                rst = np.concatenate(img_group, axis=2)
                return rst
class Stack1(object):
    """

    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):

        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        else:

            rst = np.concatenate(img_group, axis=0)
            # plt.imshow(rst[:,:,3:6])
            # plt.show()
            return torch.from_numpy(rst)

class ToTorchFormatTensor(object):
    """

    ToTorchFormatTensor

    Class for converting an image to torch tensor format.

    """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()
class ToTorchFormatTensor1(object):
    """
    Transform images in a list to PyTorch format tensors.

    Args:
        div (bool, optional): Whether to divide pixel values by 255. Default is True.

    Attributes:
        worker: Torchvision ToTensor transformation object.

    Methods:
        __call__(self, img_group): Transform a list of images to PyTorch format tensors.

    """

    def __init__(self, div=True):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class IdentityTransform(object):
    """
    A class that performs an identity transformation on the input data.

    This class does not modify the input data and returns it as is.

    Usage:
        >>> transformer = IdentityTransform()
        >>> result = transformer(data)

    Args:
        data: The input data to be transformed.

    Returns:
        The input data without any modifications.
    """
    def __call__(self, data):
        return data


# custom transforms
class GroupRandomColorJitter(object):
    """
    Applies random color jitter transformations to a group of images.

    Args:
        p (float, optional): The probability of applying the transformations. Default is 0.8.
        brightness (float, optional): How much to adjust the brightness. Default is 0.4.
        contrast (float, optional): How much to adjust the contrast. Default is 0.4.
        saturation (float, optional): How much to adjust the saturation. Default is 0.2.
        hue (float, optional): How much to adjust the hue. Default is 0.1.

    Returns:
        list: The transformed image group.

    Example:
        >>> transform = GroupRandomColorJitter()
        >>> img_group = [img1, img2, img3]
        >>> transformed_group = transform(img_group)
    """

    def __init__(self, p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.p = p
        self.worker = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img_group):

        v = random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group


class GroupRandomGrayscale(object):
    """

    """

    def __init__(self, p=0.2):
        self.p = p
        self.worker = torchvision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, img_group):

        v = random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group


class GroupGaussianBlur(object):
    """

    GroupGaussianBlur Documentation

    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random() < self.p:
            sigma = random() * 1.9 + 0.1
            return [img.filter(ImageFilter.GaussianBlur(sigma)) for img in img_group]
        else:
            return img_group


class GroupSolarization(object):
    """
    Class for solarizing a group of images.

    :param p: The probability of applying solarization to the group of images.
    :type p: float

    :return: The solarized group of images.
    :rtype: list
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random() < self.p:
            return [ImageOps.solarize(img) for img in img_group]
        else:
            return img_group