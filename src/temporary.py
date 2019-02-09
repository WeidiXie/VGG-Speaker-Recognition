import os
import sys
import numpy as np

sys.path.append('../tool')
import toolkits
import pdb

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--data_path', default='/scratch/local/ssd/weidi/voxceleb2/dev/wav', type=str)

global args
args = parser.parse_args()

# ==================================
#       Get Train/Val.
# ==================================
trnlist, trnlb = toolkits.get_voxceleb2_datalist(args, path='../meta/voxlb2_train.txt')

trn_txt = open('voxlb2_train.txt', 'w')
val_txt = open('voxlb2_val.txt', 'w')


max_lb = np.max(trnlb)+1

for i in range(max_lb):
    index = np.where(trnlb == i)[0]
    rand_index = np.random.permutation(index)
    total_index = len(rand_index)

    val_ = trnlist[rand_index[:int(total_index*0.1)]]
    trn_ = trnlist[rand_index[int(total_index*0.1):]]

    for t in trn_:
        trn_txt.write('{} {}\n'.format('/'.join(t.split(os.sep)[-3:]), i))

    for v in val_:
        val_txt.write('{} {}\n'.format('/'.join(v.split(os.sep)[-3:]), i))
