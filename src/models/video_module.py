import torch
import numpy as np
import json
import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from clip import clip
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from shutil import copy
from torch.autograd import Variable
from pytorch_lightning import LightningModule
from typing import Any
from json import dumps
from src.utils.utils import (
    check_all_rows_equal,
    round_pt_filenames,
    create_logits,
    gen_label,
    fetch_prompts,
    tokenize_sub_actions,
    LabelsManager,
    get_class_names,
    normalize_activations,
    split_video,
    text2text_similarity,
    get_peak_range,
    compute_dist_discrepancy,
    compute_average_precision_detection,
structured_prediction
)

""" """
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_distribution(scores: torch.Tensor, score_name: str, bins: int = 30):
    """
    Plot the distribution of scores using a histogram.

    :param scores: A tensor containing the scores.
    :param score_name: The name of the scores being plotted.
    :param bins: The number of bins for the histogram. Default is 30.
    :return: None
    """
    if scores.is_cuda:
        scores = scores.cpu()
    scores_np = scores.numpy()

    plt.figure(figsize=(10, 5))
    sns.histplot(scores_np, bins=bins, kde=True)
    plt.title(f'Distribution of {score_name}')  # Here is where score_name is used
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

class VideoLitModule(LightningModule):
    """
    The VideoLitModule class is a class that represents a module for video classification and analysis using multimodal embeddings. It inherits from the LightningModule class.

    Parameters:
    - model: A dictionary containing the models used for different modalities (e.g. clip, image, text). Each key represents the name of the model and the corresponding value represents the
    * model itself.
    - method: A string representing the method used for video analysis (e.g. captions, situation_localization).
    - network: A dictionary containing the configuration settings for the network.
    - decomposition: A dictionary containing the configuration settings for the decomposition of video into sub-actions.
    - loss: A dictionary containing the configuration settings for the loss function used during training.
    - solver: A dictionary containing the configuration settings for the optimizer used during training.
    - analysis: A dictionary containing the configuration settings for the analysis of video embeddings.
    - extra_args: A dictionary containing additional arguments.

    Attributes:
    - extra_args: A dictionary containing additional arguments.
    - labels_manager: An instance of the LabelsManager class for managing labels.
    - num_classes: An integer representing the number of classes.
    - dataset: A string representing the dataset used for training.
    - prompts: A list of prompts fetched based on the dataset, decomposition prompts, and input conditioning.
    - class_names: A list of class names based on the train file, dataset, and limit classes.
    - sub_action_to_encoding: A dictionary mapping sub-actions to their respective encodings.
    - clip_model: The clip model.
    - image_model: The image model.
    - text_model: The text model.
    - frame_aggregation: The frame aggregation model.
    - logit_scale: The logit scale of the clip model.
    - num: An integer representing the number of samples processed during evaluation.
    - corr_1: An integer representing the number of correct top-1 predictions during evaluation.
    - corr_5: An integer representing the number of correct top-5 predictions during evaluation.
    - correct_per_class: A list containing the number of correct predictions per class.
    - instances_per_class: A list containing the number of instances per class.
    - pred: A list containing the predicted labels.
    - gt: A list containing the ground truth labels.
    - best: A float representing the best evaluation metric achieved.
    - sub_actions_activation: A dictionary mapping sub-actions to their respective activations during decomposition.
    - sub_actions_activation_per_class: A dictionary mapping class names to activations of sub-actions during decomposition.
    - temporal_regions_per_class: A dictionary mapping class names to temporal regions during decomposition.
    - v_captions: A dictionary containing video captions when 'method' is set to 'captions'.

    Methods:
    - produce_visual_embeddings(batch): Produces visual embeddings for a batch of videos.
    - produce_image_description(batch, model, processor): Generates image descriptions for a batch of videos using a model and a processor.
    - produce_text_embeddings_situation_localization(batch): Produces text embeddings for a batch of videos during situation localization.
    - produce_text_embeddings(batch): Produces text embeddings for a batch of videos.
    - zero_shot_validation_step(batch, batch_idx): Performs a zero-shot validation step for a batch of videos during evaluation.
    - captions_validation_step(batch, batch_idx): Performs a validation step for a batch of videos using captions during evaluation.
    """
    def __init__(
        self,
        model: dict,
        method: str,
        network: dict,
        decomposition: dict,
        loss: dict,
        solver: dict,
        analysis: dict,
        extra_args: dict,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data utilities
        self.extra_args = extra_args
        self.labels_manager = LabelsManager(self.extra_args)
        self.num_classes = self.extra_args["num_classes"]
        self.dataset = self.extra_args["dataset"]

        # prompts
        self.prompts = fetch_prompts(
            self.extra_args["dataset"],self.hparams.decomposition.prompts, self.hparams.decomposition.input_conditioning
        )
        self.class_names = get_class_names(
            self.extra_args["train_file"],
            self.extra_args["dataset"],
            self.extra_args["limit_classes"],
        )
        self.sub_action_to_encoding = {}

        # models
        self.clip_model = model["clip"]
        self.image_model = model["image"]
        self.text_model = model["text"]
        self.frame_aggregation = model["frame_aggregation"]
        self.logit_scale = model["clip"].logit_scale.exp()

        # evaluation metrics
        self.num = 0
        self.corr_1 = 0
        self.corr_5 = 0
        self.correct_per_class = [0 for _ in range(self.num_classes + 1)]
        self.instances_per_class = [0 for _ in range(self.num_classes + 1)]
        self.pred = []
        self.gt = []
        self.best = 0.0

        # decomposition
        self.sub_actions_activation = {
            "sub_action_1": [0 for _ in range(self.extra_args["n_frames"])],
            "sub_action_2": [0 for _ in range(self.extra_args["n_frames"])],
            "sub_action_3": [0 for _ in range(self.extra_args["n_frames"])],
        }
        self.sub_actions_activation_per_class = {
            self.class_names[i]: {
                "sub_action_1": [0 for _ in range(self.extra_args["n_frames"])],
                "sub_action_2": [0 for _ in range(self.extra_args["n_frames"])],
                "sub_action_3": [0 for _ in range(self.extra_args["n_frames"])],
            }
            for i in range(len(self.class_names))
        }
        self.temporal_regions_per_class = {
            self.class_names[i]: {
                "sub_action_1": [],
                "sub_action_2": [],
                "sub_action_3": [],
            }
            for i in range(len(self.class_names))
        }


    def produce_visual_embeddings(self, batch):
        with torch.no_grad():
            video, label, _= batch
            video = video.view((-1, self.extra_args["n_frames"], 3) + video.size()[-2:])
            b, t, c, h, w = video.size()
            video = video.to(self.device).view(-1, c, h, w)
            frame_embeddings = self.image_model(video)
            frame_embeddings = frame_embeddings.view(b, t, -1)
            video_embedding = self.frame_aggregation(frame_embeddings)
            frame_embeddings = frame_embeddings / frame_embeddings.norm(
                dim=-1, keepdim=True
            )
            video_embedding = video_embedding / video_embedding.norm(
                dim=-1, keepdim=True
            )

        return video_embedding, frame_embeddings

    def produce_image_description(self,batch,model,processor):
        video, label, path = batch
        descriptions = []
        for v in path:
            temporary = []
            for i in range(len(label)):
                raw_image = Image.open(v[0][i]).convert("RGB")
                image = processor["eval"](raw_image).unsqueeze(0).to(self.device)
                description=model.generate({"image": image, "prompt": "Question: what action is this person performing? Answer:"})
                temporary.append(description)
            descriptions.append(temporary)

        return descriptions

    def produce_text_embeddings_situation_localization(self, batch):
        #when self.hparams.method == "situation_localization"
        text_embedding1 = []
        text_embedding2 = []
        for class_name in self.class_names:
                encodings = [
                    enc for _, enc in self.sub_action_to_encoding[class_name].items() if _!="original"
                ]


                action_text_encoding1 = encodings[0]
                action_text_encoding1 = action_text_encoding1 / action_text_encoding1.norm(
                    dim=-1, keepdim=True
                )
                text_embedding1.append(action_text_encoding1)

                action_text_encoding2 = encodings[1]
                action_text_encoding2 = action_text_encoding2 / action_text_encoding2.norm(
                    dim=-1, keepdim=True
                )
                text_embedding2.append(action_text_encoding2)


        text_embedding1 = torch.stack(text_embedding1)
        text_embedding2 = torch.stack(text_embedding2)
        return text_embedding1,text_embedding2
    def produce_text_embeddings(self, batch):
        text_embedding = []
        for class_name in self.class_names:
                encodings = [
                    enc for _, enc in self.sub_action_to_encoding[class_name].items()
                ]
                action_text_encoding = torch.stack(encodings).mean(dim=0)
                action_text_encoding = action_text_encoding / action_text_encoding.norm(
                    dim=-1, keepdim=True
                )
                text_embedding.append(action_text_encoding)
        text_embedding = torch.stack(text_embedding)
        return text_embedding

    def zero_shot_validation_step(
        self, batch: Any, batch_idx: int):
        with torch.no_grad():
            # the batch is a tuple containing the video and the label: processed_frames, record['label'], fp
            _, y, paths = batch
            video_embedding, frame_embedding = self.produce_visual_embeddings(batch)
            b = video_embedding.size(0)
            text_embedding = self.produce_text_embeddings(batch)
            similarity, _ = self.compute_similarity(
                video_embedding, text_embedding, 0, self.hparams.decomposition.use_templates,
                self.hparams.decomposition.input_conditioning)

            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)

            self.compute_metrics(indices_1, indices_5, y, b)



    def compute_metrics(self, indices_1, indices_5, y, b):
        self.num += b

        for i in range(b):
            predicted_label = indices_1[i]
            label = y[i]
            if indices_1[i].item() == label.item():
                self.corr_1 += 1
            if indices_5 is not None:
                if y[i] in indices_5[i]:
                    self.corr_5 += 1
            self.instances_per_class[label] += 1
            if label.item() == predicted_label.item():
                self.correct_per_class[label] += 1
        self.pred.extend([i[0] for i in indices_1.cpu().tolist()])
        self.gt.extend(list(y.cpu().tolist()))

    def on_validation_epoch_start(self):
        self.fill_sub_action_to_encoding()
        if self.hparams.analysis.compute_text2text_similarities:
            self.compute_text2text_similarities()
            exit()
        self.num = 0
        self.corr_1 = 0
        self.corr_5 = 0
        self.correct_per_class = [0 for _ in range(self.num_classes + 1)]
        self.instances_per_class = [0 for _ in range(self.num_classes + 1)]
        self.pred = []
        self.gt = []
        """
        prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
        """
        if self.hparams.method.endswith("localization"):
            self.localization_proposals = {i: {'video-id':[],'t-start':[],'t-end':[],'score':[]} for i in range(20)}
            self.localization_gt = {i: {'video-id': [], 't-start': [], 't-end': []} for i in range(20)}

            self.localization_mAP = {}

            self.action_classes={} #video_id: [action classes]
            txt_file ="./txt/thumos2014_localization/test_split1.txt"
            #check if the file exists
            if not os.path.exists(txt_file):
                txt_file = "../txt/thumos2014_localization/test_split1.txt"

            with open(txt_file, 'r') as file:
                for line in file:
                    #BasketballDunk/video_test_0000179  7.8 9.0 0
                    # Split the line into parts based on spaces
                    parts = line.split()

                    # Extract relevant information
                    action_class = parts[0].split('/')[0]
                    video_id = parts[0].split('/')[-1]  # Extract the video_id

                    if not video_id in self.action_classes.keys():
                        self.action_classes[video_id]=list()

                    self.action_classes[video_id].append(action_class)
                    t_start = float(parts[1])  # Extract the start time
                    t_end = float(parts[2])  # Extract the end time
                    i = int(parts[3])  # Extract the value of i

                    # Append the extracted information to the corresponding lists in the gt dictionary
                    self.localization_gt[i]['video-id'].append(video_id)
                    self.localization_gt[i]['t-start'].append(t_start)
                    self.localization_gt[i]['t-end'].append(t_end)

    def fill_sub_action_to_encoding(self):
        with torch.no_grad():
            for action in self.class_names:
                tokenized_sub_actions = tokenize_sub_actions(
                    self.prompts["prompts"],
                    self.labels_manager.label_name_to_id(action),
                    self.labels_manager,
                    include_class_name=True,
                    templates=self.hparams.decomposition.use_templates,
                    new_templates=self.hparams.decomposition.new_templates
                )
                text_inputs = tokenized_sub_actions["classes"].to(self.device)
                num_templates = tokenized_sub_actions["num_templates"]

                text_embedding = self.text_model(text_inputs)

                text_embedding = text_embedding.view(
                    num_templates, -1, 512)
                text_embedding = text_embedding / text_embedding.norm(
                    dim=-1, keepdim=True
                )
                text_embedding = text_embedding.mean(dim=0)
                text_embedding = text_embedding / text_embedding.norm(
                    dim=-1, keepdim=True
                )

                self.sub_action_to_encoding[action] = {
                    "sub_action_{}".format(i): text_embedding[i, :]
                    for i in range(1, text_embedding.size(0))
                }
                self.sub_action_to_encoding[action]["original"] = text_embedding[0, :]#original 只包含template + label
    def on_validation_epoch_end(self):
        if not self.hparams.method.endswith("localization"):
            # val_accuracy = float(self.corr_1) / self.num
            if self.num != 0:
                val_accuracy = float(self.corr_1) / self.num
            else:
                val_accuracy = 0  # Or any default value you like.
            self.log("validation_accuracy_top1", val_accuracy, sync_dist=True)

            # val_accuracy5 = float(self.corr_5) / self.num
            if self.num != 0:
                val_accuracy5 = float(self.corr_5) / self.num
            else:
                val_accuracy5 = 0
            self.log("validation_accuracy_top5", val_accuracy5, sync_dist=True)

            file_path = "{}/accuracy.txt".format(self.extra_args["output_dir"])
            with open(file_path, "a") as f:
                f.write(str(val_accuracy))
                f.write("\n")
                f.write(str(val_accuracy5))
                f.close()
            is_best = False
            if val_accuracy > self.best:
                self.best = val_accuracy
                is_best = True


            if not self.trainer.sanity_checking:

                cm = confusion_matrix(
                    self.labels_manager.convert(self.gt),
                    self.labels_manager.convert(self.pred),
                    labels=self.class_names,
                )

                names = ",".join(self.class_names)
                file_path = "{}/cm.txt".format(self.extra_args["output_dir"])
                with open(file_path, "a") as f:
                    f.write(names)

                    for i in cm:
                        j = [str(num) for num in i]
                        l = str(",".join(j) + "\n")
                        f.write(l)
                    f.close()

                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=self.class_names
                )
                #disp.plot(xticks_rotation="vertical")

                fig, ax = plt.subplots(figsize=(40, 40))
                disp.plot(ax=ax,xticks_rotation="vertical")


                plt.savefig("{}/cm_last.png".format(self.extra_args["output_dir"]))
                if is_best:
                    copy(
                        "{}/cm_last.png".format(self.extra_args["output_dir"]),
                        "{}/cm_best.png".format(self.extra_args["output_dir"]),
                    )

                self.trainer.logger.log_image(
                    key="confusion_matrix",
                    images=["{}/cm_last.png".format(self.extra_args["output_dir"])],
                )

                plt.close()
        else:
            filename = "{}/{}_{}_{}_maps.txt".format(self.extra_args["output_dir"],self.hparams.method,self.hparams.decomposition.input_conditioning,self.hparams.decomposition.use_templates)
            with open(filename, 'w') as file:
                for key, value in self.localization_mAP.items():
                    file.write(f"{key}: {value}\n")
            maps = [float(i) for i in self.localization_mAP["ave"].tolist()]
            map3, map4, map5, map6, map7 = maps
            map_ave = sum(maps) / len(maps)



            self.log("MAP3",map3, sync_dist=True)
            self.log("MAP4", map4, sync_dist=True)
            self.log("MAP5", map5, sync_dist=True)
            self.log("MAP6", map6, sync_dist=True)
            self.log("MAP7", map7, sync_dist=True)
            self.log("MAP_average", map_ave, sync_dist=True)



    def calculate_map(self,correct,proposal):
        precisions=list()
        for i in range(21):
            if proposal[i]!=0:
                p=correct[i]/proposal[i]
                precisions.append(p)
        mp=sum(precisions)/len(precisions)

        return mp

    def model_step(self, batch: Any):

        _, label = batch
        video_embedding, text_embedding = self.produce_embeddings(batch)

        # compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_video, logits_per_text = create_logits(
            video_embedding, text_embedding, logit_scale
        )

        # generate ground-truth label
        ground_truth = (
            torch.tensor(gen_label(label), dtype=video_embedding.dtype)
            .float()
            .to(self.device)
        )

        # compute loss
        loss_video = self.loss_video(logits_per_video, ground_truth)
        loss_text = self.loss_text(logits_per_text, ground_truth)
        loss = (loss_video + loss_text) / 2
        loss = loss * self.hparams.loss.weight

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            _, labels, _ = batch
            _, video_embedding = self.produce_visual_embeddings(batch)
            b, n_frames, feat_dim = video_embedding.size()
            for i in range(b):
                video = video_embedding[i, :, :]
                (
                    text_embedding,
                    num_templates,
                ) = self.produce_decomposed_actions_text_embeddings(labels[i])
                for f in range(n_frames):
                    frame = video[f, :]
                    similarity, _ = self.compute_similarity(
                        frame, text_embedding, num_templates
                    )
                    _, indices_1 = similarity.topk(1, dim=-1)
                    self.sub_actions_activation[
                        "sub_action_{}".format(indices_1.item() + 1)
                    ][f] += 1
                    self.sub_actions_activation_per_class[
                        self.labels_manager.id_to_label_name(int(labels[i]))
                    ]["sub_action_{}".format(indices_1.item() + 1)][f] += 1

        loss = Variable(
            torch.tensor(0.0, dtype=video_embedding.dtype).to(self.device),
            requires_grad=True,
        )

        # return loss or backpropagation will fail
        return {"loss": loss}

    def on_fit_start(self):
        self.fill_sub_action_to_encoding()

    def on_train_epoch_start(self):
        self.sub_actions_activation_per_class = {
            self.class_names[i]: {
                "sub_action_1": [0 for _ in range(self.extra_args["n_frames"])],
                "sub_action_2": [0 for _ in range(self.extra_args["n_frames"])],
                "sub_action_3": [0 for _ in range(self.extra_args["n_frames"])],
            }
            for i in range(len(self.class_names))
        }

    def on_train_epoch_end(self):
        if self.hparams.decomposition.video_split_mode == "dynamic":
            for action in self.sub_actions_activation_per_class:
                for j in range(self.prompts["num_steps"]):
                    dist = self.sub_actions_activation_per_class[action][
                        "sub_action_{}".format(j + 1)
                    ]
                    peak_range = get_peak_range(
                        dist, self.hparams.decomposition.similarity_ensemble.window_size
                    )
                    self.temporal_regions_per_class[action][
                        "sub_action_{}".format(j + 1)
                    ] = peak_range
        self.plot_activations()

    def plot_activations(self):
        fig, ax = plt.subplots()
        sub_actions_activation = normalize_activations(self.sub_actions_activation)
        df = pd.DataFrame(sub_actions_activation)
        p = sns.lineplot(
            data=df[[sub_action for sub_action in self.sub_actions_activation]]
        )
        p.set_ylabel("Activation")
        p.set_xlabel("Frames")
        plt.savefig("activations.png")
        self.trainer.logger.log_image(key="activations", images=["activations.png"])

        images = []
        for class_name in self.sub_actions_activation_per_class:
            fig, ax = plt.subplots()
            sub_actions_activation = normalize_activations(
                self.sub_actions_activation_per_class[class_name]
            )
            df = pd.DataFrame(sub_actions_activation)
            p = sns.lineplot(
                data=df[[sub_action for sub_action in sub_actions_activation]]
            )
            p.set_title(class_name)
            p.set_ylabel("Activation")
            p.set_xlabel("Frames")
            plt.savefig("activations_{}.png".format(class_name))
            images.append("activations_{}.png".format(class_name))

        self.trainer.logger.log_image(
            key="class_wise_activations",
            images=images,
        )

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)


    def validation_step(self, batch: Any, batch_idx: int):
        if self.hparams.method == "ensemble":
            self.ensemble_validation_step(batch, batch_idx)
        elif self.hparams.method == "structured_prediction":
            self.structured_prediction_validation_step(batch, batch_idx)
        elif self.hparams.method == "simple":
            self.zero_shot_validation_step(batch, batch_idx)



            raise ValueError("Method {} not supported".format(self.hparams.method))

    def produce_visual_embeddings_to_save(self, batch):
        with torch.no_grad():
            video = batch[0]
            num_frames = video.size(1)/3

            video = video.view((-1, int(num_frames), 3) + video.size()[-2:])
            frame_embeddings_list = []

            chunk_len = 500
            num_chunk = int(num_frames//chunk_len)
            if num_frames%chunk_len!=0:
                num_chunk+=1

            for i in range(num_chunk):
                begin=i*chunk_len
                if (i+1)*chunk_len>num_frames:
                    end = num_frames
                else:
                    end= (i+1)*chunk_len

                begin=int(begin)
                end=int(end)

                chunk_video = video[:, begin:end, :, :, :]
                chunk_size = chunk_video.size(1)
                chunk_video = chunk_video.view((-1, chunk_size, 3) + chunk_video.size()[-2:])
                b, t, c, h, w = chunk_video.size()
                chunk_video = chunk_video.to(self.device).view(-1, c, h, w)
                frame_embeddings = self.image_model(chunk_video)
                frame_embeddings = frame_embeddings.view(b, t, -1)
                frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
                frame_embeddings_list.append(frame_embeddings)

        return torch.cat(frame_embeddings_list, dim=1)


    def structured_prediction_validation_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            # fetch the batch
            _, y, _ = batch
            video_embedding, frame_embeddings = self.produce_visual_embeddings(batch)
            b = video_embedding.size(0)
            tv = []#text embedding
            for action in self.class_names:
                for j in range(self.prompts["num_steps"]):
                    tv.append(self.sub_action_to_encoding[action][
                        "sub_action_{}".format(j + 1)
                    ])

            batch_scores = []
            tv = torch.stack(tv)

            # for each video in the batch
            for i in range(b):
                ev = frame_embeddings[i, :]#embedding of one video
                score = 100.0 * ev @ tv.t()

                # apply temperature
                score = (
                    score / self.hparams.network.temperature
                    if self.hparams.network.temperature != 1.0
                    else score
                )

                action_prediction = structured_prediction(score)
                batch_scores.append(action_prediction)

            batch_scores = torch.stack(
                batch_scores) # shape of batch_scores: num of video in one batch x num of action classes

            _, top1_indices = torch.FloatTensor(batch_scores).topk(1, dim=-1)
            _, top5_indices = torch.FloatTensor(batch_scores).topk(5, dim=-1)
            self.compute_metrics(top1_indices, top5_indices, y.cpu() , b )

    def situation_localization(self,batch):
        text_embedding1,text_embedding2 = self.produce_text_embeddings_situation_localization(batch)

        with torch.no_grad():
            video_ids = []
            for item in self.localization_gt.values():
                video_ids.extend(item['video-id'])
            video_ids=list(set(video_ids))

            for id in video_ids:
                #now we look at one video
                #the same video may contains multiple classes of the action
                path = join("data", "thumos14_frames", self.action_classes[id][0],id)
                file_list = os.listdir(path)
                pt_files = [file for file in file_list if file.endswith(".pt")]
                #timestamp_41.8_frame_1254.pt
                pt_files = sorted(pt_files, key=self.sort_by_id)
                emd_list = []

                for file_name in pt_files:
                    file_path = os.path.join(path, file_name)
                    tensor = torch.load(file_path)
                    emd_list.append(tensor)

                frame_embedding = torch.cat(emd_list, dim=0).unsqueeze(0).to(text_embedding1.dtype)
                timestamps=[i.split("_")[1] for i in pt_files]

                similarity1, _ = self.compute_similarity(
                        frame_embedding, text_embedding1,0,templates=self.hparams.decomposition.use_templates,add_label=self.hparams.decomposition.input_conditioning)
                similarity1 = similarity1.squeeze(0)

                similarity2, _ = self.compute_similarity(
                        frame_embedding, text_embedding2,0,templates=self.hparams.decomposition.use_templates,add_label=self.hparams.decomposition.input_conditioning)
                similarity2 = similarity2.squeeze(0)

                self.generate_proposals_situation_localization(similarity1, similarity2,timestamps,id)
            #self.localization_proposals = {i: {'video_id':[],'t-start':[],'t-end':[],'score':[]} for i in range(20)}
            #self.localization_gt = {i: {'video_id': [], 't-start': [], 't-end': []} for i in range(20)}
            #gt is ground truth
            #self.localization_mAP = {i: [] for i in range(20)}
            for i in self.localization_proposals.keys():
                proposals = pd.DataFrame(self.localization_proposals[i])
                gt = pd.DataFrame(self.localization_gt[i])
                self.localization_mAP[i]=compute_average_precision_detection(gt,proposals)

            average = np.mean( np.vstack(self.localization_mAP.values()), axis=0)
            self.localization_mAP["ave"]=average
            self.on_validation_epoch_end()


    def sort_by_id(self, file_name):
        # return int(file_name.split('_frame_')[1].split('.pt')[0])
        # return int(file_name.split('_')[1].split('.pt')[0])
        split_name = file_name.split('_')  # split by underscore
        if len(split_name) > 1:
            split_extension = split_name[-1].split('.')  # split the last element by dot
            if len(split_extension) > 0:
                return int(float(split_extension[0]))  #



    def ensemble_validation_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():

            # fetch the batch
            _, y, _ = batch
            video_embedding, frame_embeddigs = self.produce_visual_embeddings(batch)
            b = video_embedding.size(0)

            batch_scores = []

            # for each video in the batch
            for i in range(b):
                final_scores = []

                # get video embedding
                batch_video_embedding = video_embedding[i, :]


                # for each action
                for action in self.class_names:

                    # get text embedding of the action
                    text_embedding = self.sub_action_to_encoding[action]["original"]
                    score = 100.0 * batch_video_embedding @ text_embedding.t()

                    # apply temperature
                    score = (
                        score / self.hparams.network.temperature
                        if self.hparams.network.temperature != 1.0
                        else score
                    )

                    # get text embeddings of the sub-actions
                    action_scores = [score.item()]

                    #不减一的话这里self.hparams.method == "description_split"就运行不了
                    for j in range(1,len(self.sub_action_to_encoding[action])-1):

                        if self.hparams.decomposition.video_split_mode == "uniform":
                            video_splits = split_video(frame_embeddigs,n_splits=len(self.sub_action_to_encoding[action]),mode=self.hparams.decomposition.video_split_mode,)
                            clip_video_embedding = video_splits[j][i, :]
                        elif self.hparams.decomposition.video_split_mode == "no_split":
                            clip_video_embedding = batch_video_embedding
                        else:
                            raise ValueError(
                                "Video split mode {} not supported".format(
                                    self.hparams.decomposition.video_split_mode
                                )
                            )
                        text_embedding = self.sub_action_to_encoding[action][
                            "sub_action_{}".format(j + 1)
                        ]

                        # compute similarity between current clip and corresponding sub-action
                        score = 100.0 * clip_video_embedding @ text_embedding.t()

                        # apply temperature
                        score = (
                            score / self.hparams.network.temperature
                            if self.hparams.network.temperature != 1.0
                            else score
                        )

                        action_scores.append(score.item())

                    # average scores to obtain similarity score for the whole video
                    action_scores = torch.stack(
                        [
                            torch.tensor(action_score).float()
                            for action_score in action_scores
                        ]
                    )
                    action_score = action_scores.mean()
                    final_scores.append(action_score.item())

                final_scores = torch.tensor(final_scores).to(self.device)
                batch_scores.append(final_scores)
            batch_scores = torch.stack(batch_scores)
            similarity = batch_scores.softmax(dim=-1)
            _, indices_1 = similarity.topk(1, dim=-1)
            _, indices_5 = similarity.topk(5, dim=-1)

            self.compute_metrics(indices_1, indices_5, y, b)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode_prompt(self, text):
        return self.clip_model.encode_text(self.clip_model.tokenize(text)).float()

    def compute_text2text_similarities(self):
        with open("txt2txt_similarities.txt", "w") as f:
            for action in self.sub_action_to_encoding:
                original = self.sub_action_to_encoding[action]["original"]
                for i in range(self.prompts["num_steps"]):
                    sub_action = self.sub_action_to_encoding[action][
                        "sub_action_{}".format(i + 1)
                    ]
                    f.write(
                        "sim({}, sub_task_{}) = {}\n".format(
                            action,
                            i + 1,
                            text2text_similarity(original, sub_action).item(),
                        )
                    )
                other_action_similarities = []
                for other_action in self.sub_action_to_encoding:
                    if other_action != action:
                        for i in range(self.prompts["num_steps"]):
                            other_sub_action = self.sub_action_to_encoding[
                                other_action
                            ]["sub_action_{}".format(i + 1)]
                            other_action_similarities.append(
                                text2text_similarity(original, other_sub_action).item()
                            )
                f.write(
                    "sim({}, other) = {}\n".format(
                        action,
                        np.mean(other_action_similarities),
                    )
                )
                f.write("\n")


    def produce_decomposed_actions_text_embeddings(self, label):
        with torch.no_grad():
            prompts = tokenize_sub_actions(
                self.prompts["prompts"],
                label,
                self.labels_manager,
                templates=self.hparams.decomposition.use_templates,
                new_templates=self.hparams.decomposition.new_templates
            )
            text_inputs = prompts["classes"]
            num_templates = prompts["num_templates"]
            text_inputs = text_inputs.to(self.device)
            text_embedding = self.text_model(text_inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding, num_templates


    def compute_similarity(
        self,
        video_embedding,
        text_embedding,
        num_templates,
        templates,
        add_label
    ):

        b = video_embedding.size()[0]

        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        scores =  100.0 * video_embedding @ text_embedding.t()


        scores = (
            scores / self.hparams.network.temperature
            if self.hparams.network.temperature != 1.0
            else scores
        )


        similarity = scores.softmax(dim=-1)

        return similarity, scores



    def produce_embeddings(self, batch, frame_level=False):

        # build video data
        video, label = batch
        video = video.view((-1, self.extra_args["n_frames"], 3) + video.size()[-2:])
        b, t, c, h, w = video.size()

        # build text data
        text_id = np.random.randint(self.prompts["num_steps"], size=len(label))
        texts_source = torch.stack(
            [self.prompts["text_dict"][j][i, :] for i, j in zip(label, text_id)]
        )

        # produce video embeddings
        video = video.to(self.device).view(-1, c, h, w)
        frame_embeddings = self.image_model(video)
        frame_embeddings = frame_embeddings.view(b, t, -1)
        if not frame_level:
            video_embedding = self.frame_aggregation(frame_embeddings)

        # produce text embeddings
        texts_source = texts_source.to(self.device)
        text_embedding = self.text_model(texts_source)

        # normalize embeddings
        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return video_embedding, text_embedding


if __name__ == "__main__":
    _ = VideoLitModule(None, None, None)
