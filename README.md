# Object-centric Video Representation for Long-term Action Anticipation

This is official implementation for WACV2023 paper: [Object-centric Video Representation for Long-term Action Anticipation](https://arxiv.org/pdf/2311.00180.pdf)

## Installation

If you are using OSCAR (Brown University's cluster): 

```bash
module load python/3.9.0 ffmpeg/4.0.1 gcc/10.2
```

Clone this repository.

```bash
git clone git@github.com:brown-palm/ObjectPrompt.git
cd ObjectPrompt
```

Set up Python (3.9) virtual environment. Install pytorch with the right CUDA version. 

```bash
python3 -m venv venvs/objectprompt
source venvs/objectprompt/bin/activate
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install other packages.

```bash
pip install -r requirements.txt 
```

## Dataset Preparation

### Ego4D

Download Ego4D dataset, annotations and pretrained models. Remember to set `path_to_ego4d` to the ego4D data path.

```bash
# set download dir for Ego4D
# export EGO4D_DIR=path_to_ego4d

# link data to the current project directory
mkdir -p data/ego4d/annotations/ data/ego4d/clips_hq/ data/ego4d/clips/
ln -s ${EGO4D_DIR}/v1/annotations/* data/ego4d/annotations/
ln -s ${EGO4D_DIR}/v1/clips/* data/ego4d/clips_hq/
ln -s ${EGO4D_DIR}/v1/clips_low_res/* data/ego4d/clips/

# link model files to current project directory
mkdir -p pretrained_models
ln -s ${EGO4D_DIR}/v1/lta_models/pretrained_models/* pretrained_models/
```


## Download Object-level Features

Download image-level and object-level features extracted using [GLIP](https://arxiv.org/pdf/2112.03857.pdf) and [CLIP](https://arxiv.org/pdf/2103.00020.pdf) to the following paths.

```bash
data/glip/object_manual_olcs
data/glip/image

data/clip/object_manual_olcs
data/clip/image
```
T: number of frames @1FPS. \
GLIP image-level feature size: `[T, 256]` \
GLIP object-level feature size: `[T, 10, 232]`. 10: number of objects per frame. 232 = 256 (object feature size) + 4 (object bbox: x1, y1, x2, y2) + 1 (object class ID) + 1 (confidence score) \
CLIP image-level feature size: `[T, 256]` \
CLIP object-level feature size: `[T, 10, 774]`. 774 = 768 (object feature size) + 4 + 1 + 1


## Video-only Models
### Train
```shell
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video.yaml \
--exp_name ego4d/sf_video
```
This will create log files and checkpoints in `lightning_logs/ego4d/sf_video`. You can launch tensorboard to monitor training process.

### Test
After training, please go to `lightning_logs/ego4d/sf_video` to find the best checkpoint. Set `CKPT_PATH` to the path of the best checkpoint.
```shell
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video.yaml \
--exp_name ego4d/sf_video \
train.enable False \
test.enable True \
ckpt_path CKPT_PATH 
```


## Video+Object Models
### Train
```shell
# GLIP object features:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video_image_object_glip.yaml \
--exp_name ego4d/sf_video_image_object_glip_lr1e-3_epoch30 \
data.image.base_path data/glip/image \
data.object.base_path data/glip/object_manual_olcs

# CLIP object features:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video_image_object_clip.yaml \
--exp_name ego4d/sf_video_image_object_clip_lr1e-3_epoch30 \
data.image.base_path data/clip/image \
data.object.base_path data/clip/object_manual_olcs
```
This will create log files and checkpoints in `lightning_logs/$exp_name$`. You can launch tensorboard to monitor training process.


### Test
After training, please go to `lightning_logs/$exp_name$` to find the best checkpoint. Set `CKPT_PATH` to the path of the best checkpoint.
```shell
# GLIP object features:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video_image_object_glip.yaml \
--exp_name ego4d/sf_video_image_object_glip_lr1e-3_epoch30 \
train.enable False \
test.enable True \
ckpt_path CKPT_PATH 

# CLIP object features:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video_image_object_clip.yaml \
--exp_name ego4d/sf_video_image_object_clip_lr1e-3_epoch30 \
train.enable False \
test.enable True \
ckpt_path CKPT_PATH 
```

