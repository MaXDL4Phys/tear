
# "the code will be released soon" 


# TEAR:  Text-Enhanced Zero-Shot Action Recognition


<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

<br>

## To start

```bash
# clone project
git clone https://github.com/MaXDL4Phys/tear
cd language_driven_action_recognition_localization

# install requirements
pip install -r requirements.txt
```
Then goes to https://github.com/openai/CLIP to install CLIP according to the instructions.

## Dataset Downloading and Preparation
This Python script extracts frames from videos in a dataset (like UCF101, HMDB51, or K600)
and saves them as images, while also handling directory creation and management.
### UCF101

The code is a Python command to extract frames from the 
"ucf101" dataset, specifying paths for input and output.
```
# Download the dataset from https://www.crcv.ucf.edu/data/UCF101/UCF101.rar, then unzip the file
#extract frames from videos
cd language_driven_action_recognition_localization/src/utils
python extract_frames.py --input <the path to downloaded dataset> --output <where you want to save the extracted frames> --dataset=ucf101
```
### HMDB51
The code is a Python command to extract frames from the 
"hmdb51" dataset, specifying paths for input and output.
```
#Download the dataset from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/, then unzip the file
#extract frames from videos
cd language_driven_action_recognition_localization/src/utils
python extract_frames.py --input <the path to downloaded dataset> --output <where you want to save the extracted frames> --dataset=hmdb51
```
### KINETICS 600
#### Kinetics-600 Download:

##### Clone repo and enter directory
```
git clone https://github.com/cvdfoundation/kinetics-dataset.git
cd kinetics-dataset
```
##### Download tar gzip files
This will create two directories, k600 and k600_targz. Tar gzips will be in k600_targz, you can delete k600_targz directory after extraction.
```
bash ./k600_downloader.sh
```
##### Extract tar gzip files
To extract the validation videos in folder related to classes use the script k600_video_extractor.sh. This script will extract the videos in the folder related to the classes.
```
bash ./k600_video_extractor.sh
```
##### Extract frames
```
cd language_driven_action_recognition_localization/src/utils
python extract_frames.py --input <the path to downloaded dataset> --output <where you want to save the extracted frames> --dataset=k600
#Create symbolic link to the folder where extracted images are saved at language_driven_action_recognition_localization/data
```
###### Split for k-600




### THUMOS 2014
```
#Download test set from https://www.crcv.ucf.edu/THUMOS14/download.html
cd language_driven_action_recognition_localization/src/utils
python extract_frames.py --input <the path to downloaded dataset> --output <where you want to save the extracted frames> --dataset=thumos2014_localization
```


## Experiments

### UCF101
```
python zsacr-ucf101-run.py
```
### HMDB51
```
python zsacr-hmdb51-run.py
```
### KINETICS 600
```
python zsacr-k600-run.py
```

### THUMOS 2014
```
CUDA_VISIBLE_DEVICES=<visible CUDA number> python -m src.eval experiment="thumos2014" model.decomposition.alpha=0.8 model.network.temperature=0.5 logger.wandb.offline=True logger.wandb.tags=["t_1"] logger.wandb.project=debug trainer=gpu trainer.devices=1 data.num_workers=0 model.decomposition.use_templates=<True or False> model.decomposition.prompts=<the type of prompt to use. Choose one from [only_label, description, decomposition, context, situation, newall]> model.decomposition.input_conditioning=<True or False> model.method="zero_shot_localization" data.batch_size=212
```
