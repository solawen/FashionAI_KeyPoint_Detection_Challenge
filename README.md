# FashionAI_KeyPoint_Detection_Challenge
- Code for [FashionAI KeyPoint Detection](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100068.5678.1.4ccc289bCzDJXu&raceId=231648&_lang=en_US)
- Score of this code : 5.59%
- Team rank 58/2322 at 1st round competition and 32/2322 at 2nd round competition
- Mainly base on [Cascaded Pyramid Network for Multi-Person Pose Estimation](https://arxiv.org/abs/1711.07319).

## Folder Structure
- `train`: store training images and annotations
- `model`: store trained models
- `test` : store testing images
- `src`: store some of the source code. \
`src/CPN_ONE_data_gen.py`: data generator. \
`src/CPN_ONE_model.py`: model definition.
- `predict.py` : predict keypoints from images.
- `preprocessing.py` : generate .mat file for training model.
- `train.py` : training model.

## Usage
1.Run the following command to generate .mat file
```shell
python preprocessing.py
```

2.Run the following command to train
```shell
python train.py --gpus=1
```

3.Modify the model path in predict (model_to_load), then run the following command to get predict .csv file.
```shell
python predict.py
```

## Requirements
- python (3.5.2)
- keras (2.1.5)
- numpy (1.14.2)
- opencv-python (3.4.0.12)
- pandas (0.22.0)
- scipy (1.0.1)
- tensorflow-gpu (1.2.0)