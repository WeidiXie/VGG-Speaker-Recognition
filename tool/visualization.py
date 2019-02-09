from __future__ import absolute_import
from __future__ import print_function
import os
import glob as gb
import numpy as np

import utils as ut
import config as cg
import pdb


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', dest='gn', help='input the gpue number')
parser.add_argument('-loss', dest='loss', help='which model to load')
parser.add_argument('-nt', dest='nt', help='normalization type')
parser.add_argument('-nfft', dest='nfft', help='nfft')
parser.add_argument('-spec_len', dest='spec_len', help='spectrogram length')
args = parser.parse_args()

# set basic GPU environments.
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gn)  # '0,1,2,3'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# ===========================================
#        Import Model
# ===========================================
from model_factory import vggvox_vlad_resnet_2D_v1, vggvox_vlad_resnet_2D_amsoftmax

# ==================================
#       Get Train/Val.
# ==================================
ut.session_print('Calculating test data lists...')
list1 = gb.glob('/media/weidi/2TB-2/datasets/wav/*/*/*')

'''

# ===================================
#	       Get VGGVox
# ===================================
train = False
verify = False
visualization = True

if args.loss == 'softmax':
    model = vggvox_vlad_resnet_2D_v1(input_dim=(cg.imgdims_v3[0], 100, cg.imgdims_v3[2]),
                                     num_class=cg.num_classes, train=train, verify=verify, vis=visualization)
elif args.loss == 'amsoftmax':
    model = vggvox_vlad_resnet_2D_amsoftmax(input_dim=(cg.imgdims_v3[0], cg.imgdims_v3[1], cg.imgdims_v3[2]),
                                            num_class=cg.num_classes, train=train, verify=verify, vis=visualization)

model_name = 'vggvox2_vlad_resnet_2D_{}_normtype{}_speclen{}/vggvox2_vlad_resnet_2D.h5'.format(args.loss, args.nt, args.spec_len)
model.load_weights(os.path.join(cg.model_dir, model_name), by_name=True)
print(model.summary())

ut.session_print('Start Testing.')
scores = []
label = []
test_augmentation = False
from vis.utils import utils
from vis.visualization import visualize_activation

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'center_assignment'
layer_idx = utils.find_layer_idx(model, layer_name)
clusters = 16
vis_images = []

for idx in range(clusters):
    img = visualize_activation(model, layer_idx, filter_indices=idx)
    vis_images.append(img[:,:,0])

vis_map = np.array(vis_images)
np.save('vis_map.npy', vis_map)
import pylab as plt
# Generate stitched image palette with 5 cols so we get 2 rows.
f, ax = plt.subplots(4,4)
for i in range(clusters):
    row = i / 4
    col = i % 4
    ax[row, col].imshow(vis_images[i])

plt.show()
'''

test_augmentation = False
clusters = np.load('vis_map.npy')
import pylab as plt

clusters_mu = np.mean(clusters, 1, keepdims=True)
clusters_std = np.std(clusters, 1, keepdims=True)
nm_clusters = (clusters - clusters_mu) / (clusters_std+1e-5)
import pdb

pdb.set_trace()
for c, p1 in enumerate(list1[::200]):
    print('Calculating {}.'.format(p1))
    spec = ut.load_data(p1, n_fft=int(args.nfft), train=False, normalization_type=int(args.nt), augmentation=test_augmentation)

    for s in range(0, spec.shape[1]-100, 40):
        section = np.expand_dims(spec[:, s: s+100], 0)
        diff = np.sum((section - nm_clusters)**2, axis=(1,2))
        index = np.argmin(diff)
        print('assigned cluster : {}'.format(index))
        #f, ax = plt.subplots(1,2)
        #ax[0].imshow(nm_clusters[index])
        #ax[1].imshow(section[0,:,:])
        #plt.show()
    pdb.set_trace()