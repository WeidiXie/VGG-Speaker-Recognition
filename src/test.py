import os
import numpy as np
import pdb
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import pylab as plt

'''
eers = []
for i in range(2, 6, 1):
    y_score = np.load('../results/verification_softmax_scores_len_{}_fold_1.npy'.format(i))
    y = np.load('../results/verification_softmax_gt_len_{}_fold_1.npy'.format(i))

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    eers += [eer]
'''
i = 6
y_score = np.load('../results/verification_gvlad_softmax_None_scores.npy')
y = np.load('../results/verification_gvlad_softmax_None_gt.npy')

fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

print(eer)
pdb.set_trace()
eers += [eer]

eers = np.array(eers)
#eers = eers.reshape((-1,4))
#eers_mu = np.mean(eers, 0)
#eers_var = np.var(eers, 0)
lengths = np.array([2,3,4,5,6])
pdb.set_trace()

plt.plot(lengths, eers, 'ro')
plt.plot(lengths, eers)

plt.xticks(range(2, 7, 1))
plt.xlabel('Segments Length(second)')
plt.ylabel('EER')
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
plt.savefig('../results/eer_vs_length.png')
print('The result EER is : {}.'.format(eer))
