import torch
import torchvision
import numpy as np
from os import listdir
from os.path import join
from src.data.components.augmentations import *


def is_img(f):
    """
    Check if the given file is an image file.

    :param f: The file to be checked.
    :return: True if the file is an image file (ends with .jpg, .jpeg, or .png), False otherwise.
    """
    return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg") or str(f).lower().endswith("png")


def get_indices(total_length: int, num_frames: int):
    """

    :param total_length: The total length of the data or sequence.
    :param num_frames: The number of frames or segments to divide the total length into.
    :return: An array of indices representing the central frame in each segment.

    """
    tick = total_length / num_frames
    indexes = np.array(
        [int(tick / 2.0 + tick * x) for x in range(num_frames)]
    )  # pick the central frame in each segment
    return indexes



def find_frames(video):
    """
    Find Frames

    This method is used to find frames in a given video directory.

    :param video: The path of the video directory.
    :return: A list of frames found in the video directory.

    """
    frames = []
    try:
        for f in sorted(listdir(video)):
            try:
                frame_number = int(f.split("_")[1].split(".")[0])
                frames.append(join(video, f))
            except:
                print(f"Error processing file: {f}")
    except FileNotFoundError:
        print(f"Directory not found: {video}")
    frames = [frame for frame in frames if is_img(frame)]
    return frames


class Stack(object):
    """

    This class represents a Stack object that can be used to concatenate a group of images.

    Attributes:
        roll (bool): Optional. Indicates whether the images should be rolled before concatenation.

    Methods:
        __init__(self, roll=False)
            Initialize a new Stack object.

        __call__(self, img_group)
            Concatenate a group of images according to the specified roll parameter.

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


class ToTorchFormatTensor(object):
    """

    ToTorchFormatTensor

    Class to convert images to torch format tensor.

    Attributes:
        div (bool): Flag to divide tensor values by 255.

    Methods:
        __init__(self, div=True): Initializes the ToTorchFormatTensor object.
        __call__(self, pic): Converts the input image to torch format tensor.

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


def get_augmentation(training, config):
    """Get the augmentation transformation for training or testing.

    :param training: A boolean flag indicating whether the transformation is for training or testing.
    :param config: The configuration object containing various settings.
    :return: The composed transformation that includes unique and common transformations.
    """
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = config.input_size * 256 // 224
    if training:

        unique = torchvision.transforms.Compose(
            [
                GroupMultiScaleCrop(config.input_size, [1, 0.875, 0.75, 0.66]),
                GroupRandomHorizontalFlip(),
                GroupRandomColorJitter(
                    p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                ),
                GroupRandomGrayscale(p=0.2),
                GroupGaussianBlur(p=0.0),
                GroupSolarization(p=0.0),
            ]
        )
    else:
        unique = torchvision.transforms.Compose(
            [GroupScale(scale_size), GroupCenterCrop(config.input_size)]
        )

    common = torchvision.transforms.Compose(
        [
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]
    )

    return torchvision.transforms.Compose([unique, common])


def process_path(path, node):
    """
    Replace specific parts of a given path based on the node parameter.

    :param path: The path to be processed.
    :param node: The node parameter indicating which replacements to perform.

    :return: The processed path.
    """
    if node == "slurm":
        path = path.replace("/data/gzara/", "/nfs/data_todi/datasets/")
    if node == "hpc":
        path = path.replace("/data/gzara/", "datasets/").replace(
            "/data/datasets/", "datasets/"
        )
    if node == "alderaan":
        path = path.replace("/data/gzara/", "/datadgx/gzara/").replace(
            "/data/datasets/", "/datadgx/datasets/"
        )
    return path
