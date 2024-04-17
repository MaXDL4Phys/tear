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

def multiprocess(
    func = None,
    data_var: str = "data",
    workers: int = 4,
    **kwargs,
) -> Any:
    """
    Split the work on a list between multiple processes.

    :param func: function to wrap
    :param data_var: name of the variable containing the data to split
    :param workers: number of workers to use to process the data
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


@multiprocess(data_var="data", workers=1)
def extract(**kwargs):
    data = kwargs["data"]
    print(data.__len__())

    if args.dataset=="ucf101":
        for v in tqdm(data):
            v_path = join(args.input, v)
            c = v.split("v_")[1].split("_")[0]
            target_class = join(args.output, c)
            create_if_not_exists_dir(target_class)
            target_video = join(target_class, v[:-4])
            makedirs(target_video)
            vidcap = cv2.VideoCapture(v_path)
            success, image = vidcap.read()
            count = 0
            success = True
            while success:
                frame_path = join(target_video, "frame_{}.jpg".format(count))
                if not exists(frame_path):
                    cv2.imwrite(frame_path, image)
                    if not exists(frame_path):
                        with open(frame_report_path, "a") as failed_report:
                            failed_report.write(frame_path + "\n")
                success, image = vidcap.read()
                count += 1

    elif args.dataset== "hmdb51":

        video = open("/home/CE/zhangshi/language_driven_action_recognition_localization/txt/hmdb51/all.txt").read().split("\n")
        video = [i.split(" ")[0] for i in video][:-1]
        video_dict = dict()

        for i in video:
            video_dict[i.split("/")[1]] = i.split("/")[0]
            #key is video name, value is video class
        for v in tqdm(data):
            v_path = join(args.input, v)
            print(v_path)
            video_name = v.split(".")[0]
            try:
                c = video_dict[video_name]
            except:
                continue
            target_class = join(args.output, c)
            create_if_not_exists_dir(target_class)
            target_video = join(target_class, v[:-4])
            makedirs(target_video)
            vidcap = cv2.VideoCapture(v_path)
            success, image = vidcap.read()
            count = 0
            success = True
            while success:
                frame_path = join(target_video, "frame_{}.jpg".format(count))
                if not exists(frame_path):
                    cv2.imwrite(frame_path, image)
                    if not exists(frame_path):
                        with open(frame_report_path, "a") as failed_report:
                            failed_report.write(frame_path + "\n")
                success, image = vidcap.read()
                count += 1



    elif args.dataset=="thumos2014_val":
        video = open("/home/CE/zhangshi/dissertation/language_driven_action_recognition_localization/txt/thumos2014/test_split2.txt").read().split("\n")
        video = [i.split(" ")[0] for i in video][:-1]
        video_dict = dict()

        for i in video:
            video_dict[i.split("/")[1]] = i.split("/")[0]
            #key is video name, value is video class
        for v in tqdm(data):
            v_path = join(args.input, v)
            video_name = v.split(".")[0]
            c = video_dict[video_name]
            target_class = join(args.output, c)
            create_if_not_exists_dir(target_class)
            target_video = join(target_class, v[:-4])
            makedirs(target_video)
            vidcap = cv2.VideoCapture(v_path)
            success, image = vidcap.read()
            count = 0
            success = True
            while success:
                frame_path = join(target_video, "frame_{}.jpg".format(count))
                if not exists(frame_path):
                    cv2.imwrite(frame_path, image)
                    if not exists(frame_path):
                        with open(frame_report_path, "a") as failed_report:
                            failed_report.write(frame_path + "\n")
                success, image = vidcap.read()
                count += 1

    elif args.dataset=="thumos2014_test":
        video = open("/home/CE/zhangshi/dissertation/language_driven_action_recognition_localization/txt/thumos2014/test_split1.txt").read().split("\n")
        video = [i.split(" ")[0] for i in video]

        video_dict = dict()
        for i in video:
            video_dict[i.split("/")[1]] = i.split("/")[0]
            #key is video name, value is video class
        for v in tqdm(data):
            v_path = join(args.input, v)
            video_name = v.split(".")[0]
            c = video_dict[video_name]

            target_class = join(args.output, c)
            create_if_not_exists_dir(target_class)
            target_video = join(target_class, v[:-4])
            makedirs(target_video)
            vidcap = cv2.VideoCapture(v_path)
            success, image = vidcap.read()
            count = 0
            success = True
            while success:
                frame_path = join(target_video, "frame_{}.jpg".format(count))
                if not exists(frame_path):
                    cv2.imwrite(frame_path, image)
                    if not exists(frame_path):
                        with open(frame_report_path, "a") as failed_report:
                            failed_report.write(frame_path + "\n")
                success, image = vidcap.read()
                count += 1

    elif args.dataset=="thumos2014_localization":
        video = open("/home/CE/zhangshi/language_driven_action_recognition_localization/txt/thumos2014_localization/test_split1.txt").read().split("\n")
        video = [i.split(" ")[0] for i in video]
        video_dict = dict()
        for i in video:
            # Some videos correspond to more than one label
            if i.split("/")[1] not in video_dict.keys():
                class_list = []
                class_list.append(i.split("/")[0])
                video_dict[i.split("/")[1]] = class_list
            else:
                video_dict[i.split("/")[1]].append(i.split("/")[0])
            #key is video name, value is list of video classes
        videos = video_dict.keys()
        for v in tqdm(data):
            v_path = join(args.input, v)
            video_name = v.split(".")[0]

            if video_name in videos:
                c_list = video_dict[video_name]
                c_list=list(set(c_list))
                for c in c_list:
                    target_class = join(args.output, c)
                    create_if_not_exists_dir(target_class)
                    target_video = join(target_class, v[:-4])
                    makedirs(target_video)

                    vidcap = cv2.VideoCapture(v_path)
                    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

                    success, image = vidcap.read()
                    count = 0
                    success = True
                    while success:
                        timestamp = count / frame_rate
                        #timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC)
                        #don't use this. The last frames of a video always have 0 as their timestamp. I can't figure out why.
                        #frame_path = join(target_video,"timestamp_{}_frame_{}.jpg".format((timestamp/1000),count))
                        frame_path = os.path.join(target_video,
                                                  "timestamp_{}_frame_{}.jpg".format(timestamp, count))

                        if not exists(frame_path):
                            cv2.imwrite(frame_path, image)

                            if not exists(frame_path):
                                with open(frame_report_path, "w") as failed_report:
                                    failed_report.write(frame_path + "\n")
                        success, image = vidcap.read()
                        count += 1
    elif args.dataset=="thumos2014_localization_test":
        video = open("/home/CE/zhangshi/language_driven_action_recognition_localization/txt/thumos2014_localization/test_split2.txt").read().split("\n")
        video = [i.split(" ")[0] for i in video]
        video_dict = dict()
        for i in video:
            # Some videos correspond to more than one label
            if i.split("/")[1] not in video_dict.keys():
                class_list = []
                class_list.append(i.split("/")[0])
                video_dict[i.split("/")[1]] = class_list
            else:
                video_dict[i.split("/")[1]].append(i.split("/")[0])
            #key is video name, value is list of video classes
        videos = video_dict.keys()
        for v in tqdm(data):
            v_path = join(args.input, v)
            video_name = v.split(".")[0]

            if video_name in videos:
                c_list = video_dict[video_name]
                c_list=list(set(c_list))
                for c in c_list:
                    target_class = join(args.output, c)
                    create_if_not_exists_dir(target_class)
                    target_video = join(target_class, v[:-4])
                    makedirs(target_video)

                    vidcap = cv2.VideoCapture(v_path)
                    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

                    success, image = vidcap.read()
                    count = 0
                    success = True
                    while success:
                        timestamp = count / frame_rate
                        #timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC)
                        #don't use this. The last frames of a video always have 0 as their timestamp. I can't figure out why.
                        #frame_path = join(target_video,"timestamp_{}_frame_{}.jpg".format((timestamp/1000),count))
                        frame_path = os.path.join(target_video,
                                                  "timestamp_{}_frame_{}.jpg".format(timestamp, count))

                        if not exists(frame_path):
                            cv2.imwrite(frame_path, image)

                            if not exists(frame_path):
                                with open(frame_report_path, "w") as failed_report:
                                    failed_report.write(frame_path + "\n")
                        success, image = vidcap.read()
                        count += 1



parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--dataset",type = str)
args = parser.parse_args()

frame_report_path = "failed_frames_report.txt"
if exists(frame_report_path):
    system("rm {}".format(frame_report_path))

failed = open("failed.txt", "w")

reset_dir(args.output)

videos = listdir(args.input)
print(videos.__len__())
extract(data=videos)

failed.close()