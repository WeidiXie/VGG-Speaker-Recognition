#!/bin/bash

export LD_LIBRARY_PATH=/home/ben/anaconda3/envs/entrance/lib:$LD_LIBRARY_PATH

python main.py --net resnet34s --gpu 0  --ghost_cluster 2 --vlad_cluster 8 --batch_size 16 --lr 0.001 --warmup_ratio 0.1 --optimizer adam --epochs 128 --multiprocess 8 --loss softmax --data_path /media/ben/datadrive/Zalo/voice-verification/Train-Test-Data/dataset/ --resume=/media/ben/datadrive/Software/VGG-Speaker-Recognition/model/weights_dropbox.h5
