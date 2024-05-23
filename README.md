# Real Image is All You Need: Unsupervised RealNet for Generalizable AI-Generated Image Detection

This repository is the official implementation of "Real Image is All You Need: Unsupervised RealNet for Generalizable AI-Generated Image Detection"

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
 We adopt the training set in [CNNSpot](https://github.com/peterwang512/CNNDetection). In order to save computational resources, you can generate Noise-Sensitive Real Repressentation(NSRR) in advance before training. To train the model in the paper, run this command:

 ```train
python Real_Pattern_Extractor/get_NSRR_train.py --input_dir ./TESTset/Train
python train.py --train_data_path ./NSRR/train --val_data_path ./NSRR/val
```

## Evaluation

Test set
Our test dataset can be downloaded from this [link](https://pan.baidu.com/s/1lhXEtjs5zA6I7s8iCHssKQ?pwd=real). To evaluate my model, run:

```eval
python Real_Pattern_Extractor/get_NSRR.py --input_dir ./TESTset/test
python eval.py --data_root ./NSEE/test --ckpt_path <path_to_ckpt>
```

## Pre-trained Models

You can download pretrained models here

## Acknowledgments

Our code is developed based on [DANet](https://github.com/zsyOAOA/DANet). Thanks for their sharing codes and models.
