import os
import torch
import warnings
import numpy as np
import re
from clip import clip
from importlib.util import find_spec
from typing import Callable
from omegaconf import DictConfig
from src.utils import pylogger, rich_utils
from os.path import join
from json import load
from scipy.stats import ks_2samp
from collections import Counter
from tqdm import tqdm

log = pylogger.get_pylogger(__name__)

def check_all_rows_equal(tensor):
    """
    Function to check if all rows in a tensor are equal

    Parameters:
    tensor (torch.Tensor): Input tensor

    Returns:
    bool: True if all rows are equal, False otherwise
    """

    # all rows should be equal to the first row
    rows_are_equal = torch.all(tensor.eq(tensor[0]), dim=1)

    # if all entries in rows_are_equal are True, then all rows are equal to each other
    rows_are_all_equal = rows_are_equal.all().item()

    return rows_are_all_equal

def round_pt_filenames(files):
    rounded_files = []
    for file in files:
        prefix, timestamp, _, frame = file.split("_")
        timestamp = round(float(timestamp), 2)
        rounded_file = f"{prefix}_{timestamp}_frame_{frame}"
        rounded_files.append(rounded_file)
    return rounded_files



def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def fetch_prompts(dataset, prompts, input_conditioning=False):
    if dataset=='caption':
        dataset='all'
    file_name = dataset+"_"+prompts

    # prompts_file = join("../data", "prompts", "{}.json".format(file_name))
    prompts_file = join("./data", "prompts", "{}.json".format(file_name))
    print("--------------------------------")
    print(prompts_file)

    # check if the prompts file exists
    if not os.path.exists(prompts_file):
        prompts_file = join("../data", "prompts", "{}.json".format(file_name)) # for the case of running from the root directory

    print("--------------------------------")
    print(prompts_file)

    with open(prompts_file, "r") as f:
        prompts = load(f)
    # print(prompts)
    if input_conditioning:
        for action in prompts:
            for i in range(len(prompts[action])):
                # print(f"Current action: {action}, i: {i}, Prompt: {prompts[action][i]}")
                prompts[action][i] = action +" "+ prompts[action][i]
    res = {"prompts": prompts, "num_steps": len(prompts[list(prompts.keys())[0]])}

    return res


def tokenize_sub_actions(prompts, label,
                         labels_manager, templates=True,
                         new_templates=False, include_class_name=False ):
    # num_templates = 0

    if templates==True:
        templates = [
            "a photo of action  {}",
            "a picture of action {}",
            "Human action of {}",
            " {}, an action",
            " {} this is an action",
            " {}, a photo of action",
            "Playing action of {}",
            " {}",
            "Playing a kind of action,  {}",
            "Doing a kind of action,  {}",
            "Look, the human is  {}",
            "Can you recognize the action of {}?",
            "Image classification of  {}",
            "An image of  {}",
            "The man is  {}",
            "The woman is  {}",
        ]
    elif templates==True and new_templates== True:
        templates = [
            "a photo depicting {}",
            "a scene of {}",
            "an activity involving {}",
            "engagement in {}",
            "the process of {}",
            "an instance of {}",
            "{} in action",
            "a moment of {}",
            "a person {}",
            "people engaged in {}",
            "interaction with {}",
            "a close-up of {}",
            "{} taking place",
            "a demonstration of {}",
            "capturing the moment of {}",
            "a sequence of {}",
            "performing the action of {}",
            "the art of {}",
            "the sport of {}",
            "{} as a hobby",
            "{} in competition",
            "a tutorial on {}",
            "a professional {}",
            "an amateur {}",
            "the concept of {}",
        ]

    else:
        # templates=[f"{{}}"]
        templates =  ["A video action of {}"] # template based on T3 model

    num_templates = len(templates)
    text_dict = {}

    label_name = labels_manager.id_to_label_name(label)
    label_name = label_name.replace("_", " ")

    if include_class_name:
        if label_name not in prompts:
            print(f"The key '{label_name}' was not found in the prompts dictionary.")
            print(f"The existing keys in the prompts dictionary are {prompts.keys()}")
        else:
            sub_actions = [label_name] + prompts[label_name]
    else:
        sub_actions = prompts[label_name]


    text_dict={}
    for i, txt in enumerate(templates):

        text_dict[i]=[txt.format(c) for c in sub_actions]
        # print("bla_i",text_dict[i])
        text_dict[i] = torch.cat([clip.tokenize(txt.format(c),truncate=True) for c in sub_actions])
        # print("text_dict_i",text_dict[i])


    classes = torch.cat([v for k, v in text_dict.items()])
    #{0: ['a photo of action Apply Eye Makeup', 'a photo of action Apply Eye Makeup Apply the primer and base', 'a photo of action Apply Eye Makeup Put on eyeliner and eyeshadow', 'aphoto of action Apply Eye Makeup Finish with mascara and false lashes'], 1: ['a picture of action Apply Eye Makeup', 'a picture of action Apply Eye Makeup Apply the primer and
    #base', 'a picture of action Apply Eye Makeup Put on eyeliner and eyeshadow', 'a picture of action Apply Eye Makeup Finish with mascara and false lashes']}
    res = {"classes": classes, "num_templates": num_templates, "templates": templates}
    #这个classes实际上是tokens

    return res

def structured_prediction(similarity_score,normalization=True,w1 = 1,w2 = 1,w3=1):
    #similarity score shape: n x (k x3)
    # n is the number of frames in one video
    # k is the number of action classes
    # 3 is the number of subactions of each action
    w1 = w1
    w2 = w2
    w3 = w3
    if normalization == True:
#        similarity_score = similarity_score.view(similarity_score.size(1),similarity_score.size(0))
        mean = similarity_score.mean(dim=1)
        std = similarity_score.std(dim=1)
        similarity_score = (similarity_score.view(similarity_score.size(1),similarity_score.size(0)) - mean) / std
        similarity_score = similarity_score.view(similarity_score.size(1), similarity_score.size(0))

    similarity_score = similarity_score.view(similarity_score.size(0), -1, 3) #now shape is n x k x 3

#    if normalization==True:
        #similarity_score = torch.nn.functional.normalize(similarity_score, dim=1)

    num_frames = similarity_score.size(0)
    num_subactions = similarity_score.size(2)
    action_pred = list()

    for i in range(similarity_score.size(1)):#iterate over all action labels
        frame_similarity = similarity_score[:,i,:] #similarity score of one action class with all frames
        frame_similarity.view(num_frames,num_subactions)#torch.Size([16, 3])


        all_probability = []
        #idx_all_probability = []

        begin_probs = frame_similarity[:, 0].view(num_frames)
        end_probs = frame_similarity[:,2].view(num_frames)
        mid_probs = frame_similarity[:,1].view(num_frames)
        # I'm just going to assume the number of subactions is three, since we don't have any other settings

        for j in range(num_frames-2):
            #frame_idx = {}
            sum_prob = 0
            sub1_prob = begin_probs[j]*w1
            #frame_idx["begin"] = j
            sum_prob += sub1_prob
            possible_frames = num_frames - 2 - j  # possible duration of subaction2

            for length in range(1,possible_frames+1):
                sub2_probs = mid_probs[j+1:j+length]*w2
                #frame_idx["mid"] = (j+1,j+length)
                sum_prob += sum(sub2_probs)

                sub3_prob = end_probs[j+length+1]*w3
                #frame_idx["end"] = j+length+1
                sum_prob += sub3_prob

                all_probability.append(sum_prob)
                #idx_all_probability.append(frame_idx)
        prob, indice = torch.FloatTensor(all_probability).topk(1, dim=-1)# keep the max prob as the prob of that action
        #frames = idx_all_probability[indice] #prob, indice, frames are predictions over one action label
        action_pred.append(prob)
    action_pred = torch.tensor(action_pred)
    return action_pred


class LabelsManager:
    def __init__(self, data):
        txt_dir = '/home/mbosetti/LDARL/'
        self.dataset = data["dataset"]
        if self.dataset == "hmdb51":
            train_file = open(os.path.join(txt_dir, data["train_file"]), "r")
            self.label_map = {}
            for line in train_file:
                split_line = line.split(" ")
                label_name = " ".join(split_line[0].split("/")[0].split("_"))
                label_id = int(split_line[1])
                if label_id not in self.label_map:
                    self.label_map[label_id] = label_name
        elif self.dataset == "ucf101":#also for thumos2014 when doing action recognition
            train_file = open(data["train_file"], "r")
            self.label_map = {}
            for line in train_file:
                split_line = line.split(" ")
                label_name = split_line[0].split("/")[0]
                label_name = " ".join(re.findall("[A-Z][^A-Z]*", label_name))
                label_id = int(split_line[1])
                if label_id not in self.label_map:
                    self.label_map[label_id] = label_name
        elif self.dataset == 'thumos2014':
            train_file = open(data["train_file"], "r")
            self.label_map = {}
            for line in train_file:
                label_name = line.split(" ")[0].split("/")[0]
                label_id = int(line.split(" ")[-1])
                if label_id not in self.label_map:
                    self.label_map[label_id] = label_name
        elif self.dataset == 'k600':
            train_file = open(os.path.join(txt_dir, data["train_file"]), "r")
            self.label_map = {}
            for line in train_file:
                split_line = line.split(" ")
                label_name = " ".join(split_line[0].split("/")[0].split("_"))
                label_id = int(split_line[1])
                if label_id not in self.label_map:
                    self.label_map[label_id] = label_name
        elif self.dataset == 'k600_splitted':
            train_file = open(os.path.join(txt_dir, data["train_file"]), "r")
            self.label_map = {}
            for line in train_file:
                class_name, video_path = line.lower().split("/")
                class_name = class_name.replace(" ", "_")
                video_path, start_frame, label = video_path.split(",")
                label = int(label)
                if label not in self.label_map:
                    self.label_map[label] = class_name

        else:
            raise ValueError("Dataset {} not supported".format(self.dataset))

    def id_to_label_name(self, label_id):
        label_id = int(label_id)
        return self.label_map[label_id]

    def label_name_to_id(self, label_name):
        for k, v in self.label_map.items():
            if v == label_name:
                return k
        raise ValueError("Label name {} not found".format(label_name))

    def convert(self, labels):
        return [self.label_map[label] for label in labels]

    def convert_tensor(self, labels):
        labels = labels.view(-1)
        return [self.label_map[label.item()] for label in labels]


def get_class_names(txt_file, dataset, limit_classes=-1):
    if limit_classes != -1:
        num_labels = limit_classes
    else:
        txt_dir = '/home/mbosetti/LDARL/'
        if dataset == "hmdb51":
            txt_file = os.path.join(txt_dir, txt_file)
            num_labels = len(list(set([int(line.split(" ")[-1]) for line in open(txt_file, "r")])))
        elif dataset == 'k600':
            txt_file = os.path.join(txt_dir, txt_file)
            num_labels = len(list(set([int(line.split(" ")[-1]) for line in open(txt_file, "r")])))
        elif dataset == 'k600_splitted':
            txt_file = os.path.join(txt_dir, txt_file)
            num_labels = len(list(set([int(line.split(",")[-1]) for line in open(txt_file, "r")])))


        else:
            num_labels = len(list(set([int(line.split(" ")[-1]) for line in open(txt_file, "r")])))

    class_names = ["none" for _ in range(num_labels)]
    for line in open(txt_file, "r"):
        if dataset == "hmdb51":
            class_name = " ".join(line.split(" ")[0].split("/")[0].split("_"))
            label = int(line.split(" ")[1])
        elif dataset == "ucf101":#also for thumos2014 when doing action recognition
            class_name = line.split(" ")[0].split("/")[0]
            class_name = " ".join(re.findall("[A-Z][^A-Z]*", class_name))
            label = int(line.split(" ")[1])
        elif dataset == 'thumos2014':
            class_name = line.split(" ")[0].split("/")[0]
            label = int(line.split(" ")[-1])
        elif dataset == 'k600':
            class_name = " ".join(line.split(" ")[0].split("/")[0].split("_"))
            label = int(line.split(" ")[1])
        elif dataset == 'k600_splitted':
            class_name, video_path =line.lower().split("/")
            class_name = class_name.replace(" ", "_")
            video_path, start_frame, label= video_path.split(",")
            label = int(label)
            # label = int(line.split(" ")[1])
        else:
            raise ValueError("Dataset {} not supported".format(dataset))
        if label < limit_classes or limit_classes == -1:
            if class_name not in class_names:
                #the problem here is that, when doing the localization, label does not start from zero
                #class_names[label] = class_name
                # index = next((i for i, x in enumerate(class_names) if x is "none"), "none")
                index = next((i for i, x in enumerate(class_names) if x == "none"), "none")

                # if index is not "none":
                if index != "none":
                    class_names[index] = class_name
    assert "none" not in class_names, "None class found in class names"
    return class_names


def normalize_activations_old(activations):
    print("Before normalization")
    print(activations)
    for sub_action in activations:
        if sum(activations[sub_action]) == 0 or sum(activations[sub_action]) == len(activations[sub_action]):
            continue
        activations[sub_action] = list(
            np.array(activations[sub_action])
            - np.min(np.array(activations[sub_action]))
        )
        activations[sub_action] = list(
            np.array(activations[sub_action])
            / np.max(np.array(activations[sub_action]))
        )
    print("After normalization")
    print(activations)
    return activations

def normalize_activations(activations):
    for sub_action in activations:
        if sum(activations[sub_action]) == 0:
            continue
        prev = activations[sub_action]
        activations[sub_action] = [float(i)/sum(prev) for i in prev]
    return activations


def split_video(video, n_splits=3, mode="uniform"):

    if mode == "uniform":
        split_size = int(video.shape[1] / n_splits)
        splits = []
        for i in range(n_splits):
            split = video[:, i * split_size : (i + 1) * split_size, :].mean(1)
            splits.append(split)
    else:
        raise ValueError("Mode {} not supported".format(mode))

    return splits


def text2text_similarity(text1, text2):
    with torch.no_grad():
        text_similarity = text1 @ text2.T
    return text_similarity


def get_peak_range(data, window_size=6):
    max_index = np.argmax(data)
    start_index = max(0, max_index - int(window_size / 2))
    end_index = min(len(data) - 1, max_index + int(window_size / 2))
    window_indices = range(start_index, end_index + 1)
    return list(window_indices)


def compute_dist_discrepancy(dist1, dist2, method="naive"):
    if method == "naive":
        discrepancy = 0
        for sub_action in dist1.keys():
            diff = sum([abs(a - b) for a, b in zip(dist1[sub_action], dist2[sub_action])])
            discrepancy += diff
    elif method == "ks":
        discrepancy = 0
        for sub_action in dist1.keys():
            discrepancy += ks_2samp(dist1[sub_action], dist2[sub_action])[0]
    else:
        raise ValueError("Method {} not supported".format(method))

    return discrepancy


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    target_segment = target_segment.astype(float)
    candidate_segments = candidate_segments.astype(float)

    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.3, 0.7, 5)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[['t-start', 't-end']].values,
            this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])


    return ap

def list_all_videos(directory, extension= (".avi", ".mp4")):
    """
    Function to traverse directory & subdirectories to find all video files
    """
    videos = []
    # Traverse directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if file ends with given extension
            if file.endswith(extension):
                # Get full file path
                full_path = os.path.join(root, file)
                videos.append(full_path)
    return videos

def single2double_quotes(text):
    result = str(re.search("\[(.*)\]", text).group(0))  # extract the list
    result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"TEMP", result)
    result = re.sub(r"'", '"', result)
    result = re.sub(r"TEMP", r"'", result)
    return result




def process_and_assign_id_k600(input_file, final_output_file):
    def process_action_line(line):
        line = line.replace('val/', '')  # Remove 'val/' prefix
        processed_line = ''.join(word.capitalize() for word in line.split(' '))  # Capitalize and remove spaces
        processed_line = processed_line.split(":")[0]  # Remove colon
        return 'val/' + processed_line + '/'  # Re-add 'val/' and append '/'

    action_dict = {}
    action_id = 0
    output_lines = []

    with open(input_file, 'r') as infile:
        current_action = ''
        for line in infile:
            line = line.strip()
            if not line:  # Skip blank lines
                continue
            if line.startswith('val/'):  # If it's an action descriptor line
                current_action = process_action_line(line)  # Process and update current action
                continue  # Skip further processing for action descriptor lines

            # Append current action to the video file lines
            full_line = current_action + line
            action = full_line.split('/')[1]  # Extract action name for ID mapping

            # Assign ID to action if not already done
            if action not in action_dict:
                action_dict[action] = action_id
                action_id += 1

            # Append the action ID to the line
            new_line = f"{full_line} {action_dict[action]}"
            output_lines.append(new_line)

    # Write the final output to file, without needing an intermediate output
    with open(final_output_file, 'w') as file:
        for line in output_lines:
            file.write(f"{line}\n")


# def lowercase_subfolders(parent_dir):
#     """
#     Lowercase all subfolders within each folder in the given parent directory.
#
#     Args:
#     - parent_dir (str): Path to the parent directory containing folders.
#     """
#     # Iterate over each folder in the parent directory
#     for root, dirs, _ in os.walk(parent_dir):
#         for dir_name in dirs:
#             # Construct the full path of the folder
#             folder_path = os.path.join(root, dir_name)
#             # Iterate over each subfolder in the current folder
#             for subfolder_name in os.listdir(folder_path):
#                 # Construct the full path of the subfolder
#                 subfolder_path = os.path.join(folder_path, subfolder_name)
#                 # Construct the new lowercase name for the subfolder
#                 new_subfolder_name = subfolder_name.lower()
#                 # Construct the new full path of the subfolder with the lowercase name
#                 new_subfolder_path = os.path.join(folder_path, new_subfolder_name)
#                 # Rename the subfolder to lowercase
#                 os.rename(subfolder_path, new_subfolder_path)
#                 print(f'Renamed: {subfolder_path} -> {new_subfolder_path}')
def lowercase_subfolders(parent_dir):
    """
    Lowercase all subfolders within each folder in the given parent directory.

    Args:
    - parent_dir (str): Path to the parent directory containing folders.
    """
    # Get a list of all directories
    all_dirs = [os.path.join(root, dir_name)
                for root, dirs, _ in os.walk(parent_dir)
                for dir_name in dirs]

    # Create a progress bar
    with tqdm(total=len(all_dirs), desc="Processing directories", ncols=100) as pbar:
        # Iterate over each directory
        for folder_path in all_dirs:
            # Iterate over each subfolder in the current folder
            for subfolder_name in os.listdir(folder_path):
                # Construct the full path of the subfolder
                subfolder_path = os.path.join(folder_path, subfolder_name)
                # Construct the new lowercase name for the subfolder
                new_subfolder_name = subfolder_name.lower()
                # Construct the new full path of the subfolder with the lowercase name
                new_subfolder_path = os.path.join(folder_path, new_subfolder_name)
                # Rename the subfolder to lowercase
                os.rename(subfolder_path, new_subfolder_path)
                print(f'Renamed: {subfolder_path} -> {new_subfolder_path}')

            # Update the progress bar
            pbar.update()

def find_duplicates(filename):
    print(f'Reading the file {filename}')
    with open(filename, 'r') as file:
        lines = file.readlines()

    unique_lines = set(lines)
    if len(lines) != len(unique_lines):
        print('Duplicate lines found:')
        duplicates = set([line for line in lines if lines.count(line) > 1])
        for duplicate in duplicates:
            print(duplicate)
    else:
        print('No duplicate lines found.')

def find_unique_classes(filename):
    print(f'Reading the file {filename}')
    with open(filename, 'r') as file:
        lines = file.readlines()

    unique_lines = set(lines)
    unique_classes = set([line.split(' ')[1] for line in unique_lines])
    print(f'Unique classes: {unique_classes}')