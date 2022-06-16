# This is a repository for implementing four basic self-supervised learning methods.

## Introduction
**This is an implementation code written in Python (version 3.6.9) for the following self-supervised learning (SSL): <br>
1) 'A Simple Framework for Contrastive Learning of Visual Representations'
   ([paper](https://arxiv.org/pdf/2002.05709.pdf))
2) 'Momentum Contrast for Unsupervised Visual Representation Learning'
   ([paper](https://arxiv.org/pdf/2003.04297.pdf))
3) 'Unsupervised Learning of Visual Features by Contrasting Cluster Assignments' ([paper](https://arxiv.org/pdf/2006.09882))
4) 'Bootstrap your own latent: A new approach to self-supervised Learning' ([paper](https://arxiv.org/pdf/2006.07733))


## Usage

All hyper-parameters apply across all datasets (default setup/experiment) in the submission document as following:


Settings related with the dataset (example of multi-crop) \
--dataset C10 # Dataset \
--data_path './' # Path of the dataset \
--nmb_workers 8 # Number of workers \
--nmb_crops 2 4 # Number of crops \ 
--size_crops 32 16 # Crop size \
--min_scale_crops 0.2 0.008 # Minimum scale of crops \
--max_scale_crops 1. 0.4 # Maximum scale of crops \
--batch_size  512 # Batch size


Settings related with the multi-crop \
--temp 0.1 # Temperature parameter apply only for SIMCLR, MoCo and SwAV \
--eps 0.05 # Epsilon parameter of SwAV \ 
--mem_bank_n # Number of batch to store applies only in MoCo \
--h 1 # Number of hidden layers in projection module \
--epochs 500 # Number of training epochs \
--encoder_mom 0.99 # Learning rate of the momentum encoder (applies only on MoCo and SwAV) \
--loss_type nce # Loss type, list: nce (SimCLR), moco (MoCo v2), swav (SwAV), byol (BYOL) \
--model_type resnet18 # Type of ResNet (available: resnet18, resnet34 and resnet50) \
--batch 512 # Training batch size \
--project_dim 128 # Dimension of project head \
--prototypes 1000 # Number of prototypes (applies only on SwAV) \


Settings related to learning process \
--lr 0.1 0.1 0.1 0.1 0.1 # Learning rates, first for CNN, second for batch layers, third for mlp, fourth for linear classification layer, and last for the testing clc layer \
--wd 1e-6 1e-6 1e-6 \ Weight decay, first for CNN layers, second for batch layers, third for MLP layers #
--warmup 10 # Warmup epoch \
--amp # if applies, it is used Mixamp (float16) forward parser \
--larc # if applies, it is used LARS optimizer instead of SGD \
--classifiers # trains only classifier layer

Settings related to the model storage and GPU <br>
--output_dir ./runs # path to store the model during the training \
--cpt_load_path ./ # if applies, path to load a stored model/settings \
--cpt_name model.cpt # File name to store the model \
--run_name default_run # Log file name \
--dev 0 # Set the GPU device (if applies)
