import torch.utils.data as data
from src.data.components.data_utils import get_indices, find_frames, process_path
from PIL import Image
from os.path import join
import os


class VideoDataset(data.Dataset):
    """
    VideoDataset class

    This class is a PyTorch Dataset implementation for loading video datasets.

    Parameters:
    - video_list (str): The path to a text file containing a list of videos.
    - dataset (str): The name of the dataset.
    - num_frames (int): The number of frames to sample from each video.
    - transform (callable, optional): A function/transform to apply to the frames.
    - limit_classes (int, optional): The limit of the classes to include in the dataset.

    Attributes:
    - num_frames (int): The number of frames to sample from each video.
    - transform (callable): A function/transform to apply to the frames.
    - video_list (list): A list of dictionaries representing the videos in the dataset.

    Methods:
    - __getitem__(self, index): Retrieves the frames, label, and file paths for a specific video index.
    - __len__(self): Returns the number of videos in the dataset.

    """
    def __init__(self, video_list, dataset, num_frames, transform=None, limit_classes=-1):

        data_dir = '/data/mbosetti'
        txt_dir = '/home/mbosetti/LDARL/'

        self.num_frames = num_frames
        # transform
        self.transform = transform

        # video_list is a list of tuples (video_id, label, path)
        self.video_list = []
        with open(os.path.join(txt_dir, video_list), 'r') as f:
            count_no = 0
            count_yes = 0
            for i, line in enumerate(f):
                if dataset == 'k600_splitted':
                    class_name, video_path = line.lower().split("/")
                    class_name = class_name.replace(" ", "_")
                    video_path, start_frame, label = video_path.split(",")
                    path = join(class_name, video_path)
                    label = int(label)
                elif dataset == 'k600_split':
                    pass
                else:
                    path, label = line.split()
                if limit_classes != -1 and int(label) >= limit_classes:
                    continue
                if dataset == 'k600_splitted':
                    path_dir = join(data_dir, 'k600')
                    # search in the subfolder["test", "train", "val"]
                    for subfolder in ["test", "train", "val"]:
                        paths = join(path_dir, subfolder, path)
                        # check if the path exists
                        if not os.path.exists(paths):
                            pass
                            # print("Path does not exist: ", paths)
                            # count_no += 1
                        else:
                            # print("Path exists: ", paths)
                            count_yes += 1
                            path = paths

                else:
                    path = join(data_dir, dataset, path)
                #check if the path exists
                if not os.path.exists(path):
                    print("Path does not exist: ", path)
                    count_no += 1
                else:
                    count_yes += 1

                record = {
                    'video_id': i,
                    'label': int(label),
                    'path': path
                }
                self.video_list.append(record)
            print("number of paths that do not exist: ", count_no)
            print("number of paths that exist: ", count_yes)

    def __getitem__(self, index):

        record = self.video_list[index]
        #check if the path exists
        # check_path(record['path'])

        # print("record path: ", record['path'])
        frame_paths = find_frames(record['path'])
        total_frames = len(frame_paths)
        if total_frames == 0:
            print(f"No frames found at {record['path']} for index {index}")
            return None, None, None  # Or some placeholder value
        indices = get_indices(total_frames, self.num_frames)

        frames = []
        fp = []
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)

            # check if frame_paths is empty
            # assert len(frame_paths) > 0, "No frames found"

            # check if indices are valid
            # assert max(indices) < len(frame_paths), "Indices exceed total frames"
            if p >= len(frame_paths):
                print("Index out of range, skipping this index")
                continue

            try:
                # print("frame path: ", frame_paths[p])
                # print(f"index {p} in frame_paths {frame_paths} with length {len(frame_paths)}")
                # print(f"index {p}")
                # print(f"length of frame_paths: {len(frame_paths)}")
                # assert len(frame_paths) > p,\
                #     f'frame path {frame_paths} Index {p} out of range. Length of frame_paths is {len(frame_paths)}'
                frame = Image.open(frame_paths[p]).convert("RGB")

                # print(frame)
            # except IndexError as e:
            #     print(f"Error accessing index {p} in frame_paths with length {len(frame_paths)}")
            except OSError:
                print('ERROR: Could not read frame from "{}"'.format(record["path"]))
                print("invalid indices: {}".format(indices))
                raise

            frames.append(frame)
            fp.append(frame_paths[p])

        if not frames:
            print(f"No frames could be opened at {record['path']} for index {index}")
            return None, None, None  # Or some placeholder value

        processed_frames = self.transform(frames)

        return processed_frames, record['label'], fp

    def __len__(self):
        return len(self.video_list)


def get_valid_indices(max_value, desired_quantity):
    """
    :param max_value: The maximum value to consider when generating the indices.
    :param desired_quantity: The desired quantity of indices to generate.
    :return: A list of valid indices ranging from 0 up to the minimum value between `max_value` and `desired_quantity`.
    """
    return list(range(min(max_value, desired_quantity)))

def check_path(path):
    """
    Check if the given path exists and whether it is empty or not.

    :param path: The path to check.
    :type path: str
    :return: None
    :rtype: None
    """
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
    if not os.listdir(path):
        print(f"Directory is empty: {path}")
    else:
        print(f"Directory is not empty: {path}")