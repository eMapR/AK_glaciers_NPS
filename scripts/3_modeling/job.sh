#!/bin/bash
#
# Install Pytorch 1.7 with CUDA 11.0
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
#
# Install Pytorch Lightning
# conda install pytorch-lightning -c conda-forge
#
# Install the CRFSEG package for the CRFs
# pip install crfseg
python 1_train_model.py --train_dir /mnt/d/glacier_tensors/training_data --val_dir /mnt/d/glacier_tensors/validation_data --batch_size 16