



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
git clone https://github.com/MaXDL4Phys/tear.git
cd tear
```
```
# create envirnmente and install requirements

conda create --name tear python=3.9
conda activate tear
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



## Experiments

### UCF101
```
python scripts/zsacr-ucf101-run.py
```
### HMDB51
```
python scripts/zsacr-hmdb51-run.py
```
#### HMDB51 Llama
```
python scripts/zsacr-hmdb51-run.py
```


### KINETICS 600
```
python scripts/zsacr-k600-run.py
```
