# README #

This repo will contain the code for ICASSP 2019, speaker identifcation (http://www.robots.ox.ac.uk/~vgg/research/speakerID/).

This repo contains a Keras implementation of the paper,     
[Utterance-level Aggregation For Speaker Recognition In The Wild (Xie et al., ICASSP 2019) (Oral)](https://arxiv.org/pdf/1902.10107.pdf).

### Dependencies
- [Python 2.7.15](https://www.continuum.io/downloads)
- [Keras 2.2.4](https://keras.io/)
- [Tensorflow 1.8.0](https://www.tensorflow.org/)

### Data
The dataset used for the experiments are

- [Voxceleb1, Voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

### Training the model
To train the model on the Voxceleb2 dataset, you can run

- python main.py --net resnet34s --batch_size 160 --gpu 2,3 --lr 0.001 --warmup_ratio 0.1 --optimizer adam --epochs 128 --multiprocess 8 --loss softmax --data_path ../path_to_voxceleb2

### Model 
- Models will be updated 

      google drive: https://drive.google.com/open?id=1M_SXoW1ceKm3LghItY2ENKKUn3cWYfZm

      dropbox: https://www.dropbox.com/sh/n96ekf7ilsvkjdp/AACXKDesS2ju5rp6Cyxh2PCva?dl=0
      
- Download the models and put it in the folder, model/

### Testing the model
To test a specific model on the voxceleb1 dataset, 
for example, the model trained with ResNet34s trained by adam with softmax, and feature dimension 512

- python predict.py --gpu 1 --net resnet34s --ghost_cluster 2 --vlad_cluster 8 --loss softmax --resume ../model/gvlad_softmax/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5 

- Expected results: 

        VoxCeleb1-Test: 3.22        VoxCeleb1-Test-Cleaned: 3.24
        VoxCeleb1-Test-E: 3.24      VoxCeleb1-Test-E-Cleaned: 3.13
        VoxCeleb1-Test-H: 5.17      VoxCeleb1-Test-H-Cleaned: 5.06

### Licence
The code and mode are available to download for commercial/research purposes under a Creative Commons Attribution-ShareAlike 4.0 International License(https://creativecommons.org/licenses/by/4.0/).

      Downloading this code implies agreement to follow the same conditions for any modification 
      and/or re-distribution of the dataset in any form.

      Additionally any entity using this code agrees to the following conditions:

      THIS CODE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
      IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
      TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
      PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
      HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
      EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
      PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
      LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
      NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
      SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

      Please cite the papers below if you make use of the dataset and code.

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
