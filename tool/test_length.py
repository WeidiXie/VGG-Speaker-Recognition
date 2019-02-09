from __future__ import absolute_import
from __future__ import print_function
import os
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
parser.add_argument('-test_length', dest='test_length', help='test length')
parser.add_argument('-test_fold', dest='test_fold', help='test fold')

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
'''
# ==================================
#       Get Verification List
# ==================================
ut.session_print('Calculating test data lists...')
data_path = np.load('../meta/audiopaths.npy')
data_leng = np.load('../meta/audiolengths.npy')

wav_length = 6

data_path_5s = np.array(['/'.join(str(data_path[i]).split(os.sep)[1:]) for i in np.where(data_leng > wav_length)[0]])
data_leng_5s = np.array([data_leng[i] for i in np.where(data_leng>wav_length)[0]])

data_path_5s_id = np.array([str(data_path[i]).split(os.sep)[1] for i in np.where(data_leng > wav_length)[0]])

data_path_5s_clips = np.array(['/'.join(str(data_path[i]).split(os.sep)[1:-1]) for i in np.where(data_leng > wav_length)[0]])

uq_data_path_5s_id = np.unique(data_path_5s_id)
uq_data_path_5s_clips = np.unique(data_path_5s_clips)
pdb.set_trace()

f = open('voxceleb1_length_test.txt', 'w')
total_length = len(data_path_5s)

for uid in uq_data_path_5s_id:
    index = np.where(data_path_5s_id == uid)[0]
    for p in range(100):
        ind1 = np.random.choice(index)
        ind2 = np.random.choice(index)
        clip1 = data_path_5s_clips[ind1]
        clip2 = data_path_5s_clips[ind2]
        while(clip1 == clip2):
            ind1 = np.random.choice(index)
            ind2 = np.random.choice(index)
            clip1 = data_path_5s_clips[ind1]
            clip2 = data_path_5s_clips[ind2]
        f.write('1 {} {} len1:{} len2:{}\n'.format(data_path_5s[ind1], data_path_5s[ind2],
                                                 data_leng_5s[ind1], data_leng_5s[ind2]))
    for n in range(100):
        ind1 = np.random.choice(index)
        ind2 = np.random.randint(total_length)
        clip1 = data_path_5s_id[ind1]
        clip2 = data_path_5s_id[ind2]

        while (clip1 == clip2):
            ind1 = np.random.choice(index)
            ind2 = np.random.randint(total_length)
            clip1 = data_path_5s_id[ind1]
            clip2 = data_path_5s_id[ind2]
        f.write('0 {} {} len1:{} len2:{}\n'.format(data_path_5s[ind1], data_path_5s[ind2],
                                                 data_leng_5s[ind1], data_leng_5s[ind2]))
pdb.set_trace()                                                 
f.close()
'''

verify_list = np.loadtxt(os.path.join(cg.meta_data, 'voxceleb1_length_test.txt'), str)
verify_lb = np.array([int(i[0]) for i in verify_list])
list1 = np.array([os.path.join(cg.data_path_test_length,i[1]) for i in verify_list])
list2 = np.array([os.path.join(cg.data_path_test_length,i[2]) for i in verify_list])

# ===================================
#	       Get VGGVox
# ===================================
train = False
verify = True
visualization = False
test_augmentation = False
tst_length = int(args.test_length) * 100

if args.loss == 'softmax':
    model = vggvox_vlad_resnet_2D_v1(input_dim=(cg.imgdims_v3[0], tst_length, cg.imgdims_v3[2]),
                                     num_class=cg.num_classes, train=train, verify=verify, vis=visualization)
elif args.loss == 'amsoftmax':
    model = vggvox_vlad_resnet_2D_amsoftmax(input_dim=(cg.imgdims_v3[0], tst_length, cg.imgdims_v3[2]),
                                            num_class=cg.num_classes, train=train, verify=verify, vis=visualization)

model_name = 'vggvox2_vlad_resnet_2D_{}_normtype{}_speclen{}/vggvox2_vlad_resnet_2D.h5'.format(args.loss, args.nt, args.spec_len)
model.load_weights(os.path.join(cg.model_dir, model_name), by_name=True)
print(model.summary())

ut.session_print('Start Testing.')
scores = []
label = []

for c, (p1, p2) in enumerate(zip(list1, list2)):

    if c % 200 == 0:
        print('='*50)
        print('Finish Pair: {}.'.format(c))

    spec1 = ut.load_data(p1, n_fft=int(args.nfft), train=True, normalization_type=int(args.nt), augmentation=test_augmentation, spec_len=tst_length)
    spec2 = ut.load_data(p2, n_fft=int(args.nfft), train=True, normalization_type=int(args.nt), augmentation=test_augmentation, spec_len=tst_length)

    spec1 = np.expand_dims(spec1, -1)
    spec2 = np.expand_dims(spec2, -1)

    specs = np.stack((spec1, spec2), axis=0)
    vs = model.predict(specs)

    scores += [np.sum(vs[0]*vs[1])]
    label += [verify_lb[c]]
    print('scores : {}, gt : {}'.format(scores[-1], label[-1]))

scores = np.array(scores)
label = np.array(label)
np.save('../results/verification_{}_scores_len_{}_fold_{}.npy'.format(args.loss, args.test_length, args.test_fold), scores)
np.save('../results/verification_{}_gt_len_{}_fold_{}.npy'.format(args.loss, args.test_length, args.test_fold), label)
