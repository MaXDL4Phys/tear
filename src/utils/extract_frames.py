#!/usr/bin/env bash

from argparse import ArgumentParser
from os import listdir, system, makedirs
from os.path import join, exists
import os
from tqdm import tqdm
import cv2
import multiprocessing
from functools import partial, wraps
from typing import Any, Callable, Optional
from datetime import datetime
from src.utils import list_all_videos
def multiprocess(
    func = None,
    data_var: str = "data",
    workers: int = 4,
    **kwargs,
) -> Any:
    """
    Multiprocess method that distributes data processing across multiple processes.

    Args:
        func: The function to be executed in parallel. Must accept the same arguments as `inner` function.
        data_var: The name of the keyword argument that contains the data. Defaults to "data".
        workers: The number of processes to be created. Defaults to 4.
        **kwargs: Additional keyword arguments to be passed to the `func` function.

    Returns:
        The decorated `inner` function or the result of the `func` function.

    Raises:
        ValueError: If `data_var` is not present in `**kwargs`.
        AssertionError: If `data_items` is not of type list.

    """
    if func is None:
        return partial(multiprocess, data_var=data_var, workers=workers, **kwargs)

    @wraps(func)
    def inner(*args, **inner_kwargs):
        if data_var not in inner_kwargs:
            raise ValueError(f"{data_var} is not in **kwargs")

        data_items = inner_kwargs[data_var]
        assert isinstance(data_items, (list)), "`data_items` must be of type list"

        # Evaluate the number of samples per process.
        size = len(data_items)
        item_residual = size % workers
        items_per_process = (size - item_residual) / workers
        items_per_process = int(items_per_process)

        # Instantiate the processes with their custom arguments.
        if len(data_items) > 1:
            processes = []
            for i in range(workers):
                # Get the partition of data for the current process.
                start_index = items_per_process * i
                end_index = start_index + items_per_process
                sub = data_items[start_index:end_index]

                # Copy the kwargs and overwrite the data variable.
                process_kwargs = inner_kwargs.copy()
                process_kwargs[data_var] = sub

                # Store the new process.
                processes.append(
                    multiprocessing.Process(
                        target=func, args=args, kwargs=process_kwargs
                    )
                )

            # Start each process.
            for process in processes:
                process.start()

            # Wait for each process to complete.
            for process in processes:
                process.join()

        # If necessary, create another process for the residual data.
        if item_residual != 0:
            start_index = items_per_process * workers
            sub = data_items[start_index:]
            process_kwargs = inner_kwargs.copy()
            process_kwargs[data_var] = sub

            assert func is not None, "`func` must be not None"
            func(*args, **process_kwargs)

    return inner


def reset_dir(path):
    if exists(path):
        system("rm -r {}".format(path))
    makedirs(path)


def create_if_not_exists_dir(path):
    if not exists(path):
        makedirs(path)


def get_class(path):
    path_video_name = path.split("/")[-1]
    with open("kinetics_sports_msda_train.txt", "r") as f:
        for line in f:
            video_path = line.split()[-1]
            video_name = video_path.split("/")[-1]
            if video_name == path_video_name:
                class_name = video_path.split("/")[1]
                return class_name


def create_if_not_exists_dir(path):
    if not exists(path):
        os.makedirs(path)

def extract_frames(v_path, target_video, frame_report_path):
    vidcap = cv2.VideoCapture(v_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = join(target_video, f"frame_{count}.jpg")
        print(frame_path)
        if exists(frame_path):
            print(f"Frame {frame_path} already exists")
        if not exists(frame_path):
            cv2.imwrite(frame_path, image)
            if not exists(frame_path):
                with open(frame_report_path, "a") as failed_report:
                    failed_report.write(frame_path + "\n")
        success, image = vidcap.read()
        count += 1
# Path to the combined unique identifiers file
combined_file = "/home/mbosetti/LDARL/txt/k600_split/k600_split_concat_unique.txt"

def process_dataset(data, input_dir, output_dir, dataset_func, txt_file ):
    for v in tqdm(data):
        # print(v)
        v_path = join(input_dir, v)
        # print(v_path)
        if args.dataset == "hmdb51" or args.dataset == "k600" or args.dataset == "k600_split":
            target_video = os.path.splitext(os.path.basename(v_path))[0]

            if args.dataset == "k600_split":
                # extract the targhet clasff from the txt file
                with open(combined_file, "r") as f:
                    for line in f:
                        if target_video in line:
                            target_class = line.split("/")[0]
                            #change" " wiht "_"
                            target_class = target_class.replace(" ", "_")
                            target_class = os.path.join(output_dir, target_class)
                            target_video = os.path.join(target_class, target_video)

            else:
                parent_dir = os.path.dirname(v_path)
            # Then, we get the last part of the parent directory path
                target_class = os.path.basename(parent_dir)
                target_class = os.path.join(output_dir, target_class)
                target_video = os.path.join(target_class, target_video)
        elif args.dataset == "ucf101":
            target_class, target_video = dataset_func(v)
        # elif args.dataset == "k600":
        #     target_class, target_video = dataset_func(v)

        elif not target_class or not target_video:
            continue  # Skip if class or video path is not resolved

        if exists(target_video):
            print(f"Video {target_video} already exists")
            continue
        else:
            create_if_not_exists_dir(target_class)
            create_if_not_exists_dir(target_video)
            print(target_video)
            extract_frames(v_path, target_video, frame_report_path)

@multiprocess(data_var="data", workers=1)
def extract(**kwargs):
    data = kwargs["data"]

    def ucf101_processor(v):
        c = v.split("v_")[1].split("_")[0]
        target_class = join(args.output, c)
        target_video = join(target_class, v[:-4])
        return target_class, target_video

    def hmdb51_processor(v, hmdb51_file_path):
        # Assuming hmdb51_file_path is defined and the file exists
        video_dict = {i.split("/")[1]: i.split("/")[0] for i in open(hmdb51_file_path).read().split("\n") if i}
        video_name = v.split(".")[0]
        c = video_dict.get(video_name)
        if not c:
            return None, None
        target_class = join(args.output, c)
        target_video = join(target_class, v[:-4])
        return target_class, target_video

    # Define processors for other datasets similar to ucf101_processor and hmdb51_processor

    def k600_processor(v):
        c = get_class(v)
        if not c:
            return None, None
        target_class = join(args.output, c)
        target_video = join(target_class, v[:-4])
        return target_class, target_video

    def k600_splitted_processor(v, txt_file):
        c = get_class(v)
        if not c:
            return None, None
        target_class = join(args.output, c)
        target_video = join(target_class, v[:-4])
        return target_class, target_video

    if args.dataset == "ucf101":
        process_dataset(data, args.input, args.output, ucf101_processor)
    elif args.dataset == "hmdb51":
        process_dataset(data, args.input, args.output, hmdb51_processor)
    # Add other dataset conditions
    elif args.dataset == "k600":
        process_dataset(data, args.input, args.output, k600_processor)
    elif args.dataset == "k600_split":
        process_dataset(data, args.input, args.output, k600_processor, combined_file)

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--dataset",type = str)
args = parser.parse_args()

frame_report_path = "failed_frames_report.txt"
if exists(frame_report_path):
    system("rm {}".format(frame_report_path))

failed = open("failed.txt", "w")

# reset_dir(args.output)

if args.dataset == "hmdb51" or args.dataset == "k600" or args.dataset == "k600_split":
    videos = list_all_videos(args.input)
else:
    videos = listdir(args.input)
# print(videos)

extract(data=videos)

failed.close()


