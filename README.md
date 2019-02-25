# README #

This repo will contain the code for ICASSP 2019, speaker identifcation (http://www.robots.ox.ac.uk/~vgg/research/speakerID/).

This repo contains a Keras implementation of the paper,     
[Utterance-level Aggregation For Speaker Recognition In The Wild (Xie et al., ICASSP 2019) (Oral)](http://www.robots.ox.ac.uk/~vgg/publications/2019/Xie19a/xie19a.pdf).

### Dependencies
- [Python 2.7.15](https://www.continuum.io/downloads)
- [Keras 2.2.4](https://keras.io/)
- [Tensorflow 1.8.0](https://www.tensorflow.org/)

### Data
The dataset used for the experiments are

- [Voxceleb1, Voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

### Training the model
To train the model on the Voxceleb2 dataset, you can run

- python src/main.py --net resnet34s --batch_size 160 --gpu 2,3 --lr 0.001 --warmup_raito 0.1 --optimizer adam --epochs 128 --multiprocess 8 --loss softmax --data_path ../path_to_voxceleb2

### Model 
- Models will be updated 

      google drive: https://drive.google.com/open?id=1M_SXoW1ceKm3LghItY2ENKKUn3cWYfZm

      dropbox: https://www.dropbox.com/sh/n96ekf7ilsvkjdp/AACXKDesS2ju5rp6Cyxh2PCva?dl=0
      
- Download the models and put it in the folder, model/

### Testing the model
To test a specific model on the voxceleb1 dataset, 
for example, the model trained with ResNet34s trained by adam with softmax, and feature dimension 512

- python src/predict.py --gpu 1 --net resnet34s --ghost_cluster 2 --vlad_cluster 8 --loss softmax --resume ../model/gvlad_softmax/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5 

- Expected results: 

        VoxCeleb1-Test: 3.22        VoxCeleb1-Test-Cleaned: 3.24
        VoxCeleb1-Test-E: 3.24      VoxCeleb1-Test-E-Cleaned: 3.13
        VoxCeleb1-Test-H: 5.17      VoxCeleb1-Test-H-Cleaned: 5.06

### Citation
```
@InProceedings{Xie19,
  author       = "W. Xie, A. Nagrani, J. S. Chung, A. Zisserman ",
  title        = "Utterance-level Aggregation For Speaker Recognition In The Wild.",
  booktitle    = "ICASSP, 2019",
  year         = "2019",
}

@InProceedings{Chung18,
  author       = "J. S. Chung*, A. Nagrani*, A. Zisserman ",
  title        = "VoxCeleb2: Deep Speaker Recognition.",
  booktitle    = "INTERSPEECH, 2018",
  year         = "2018",
}

@InProceedings{Nagrani17,
  author       = "A. Nagrani*, J. S. Chung*, A. Zisserman ",
  title        = "VoxCeleb: A Large-scale Speaker Identification Dataset.",
  booktitle    = "INTERSPEECH, 2017",
  year         = "2018",
}
```


