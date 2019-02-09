from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np

sys.path.append('../tool')
import toolkits
import utils as ut

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model

    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))

    if args.test_type == 'normal':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test.txt', str)
    elif args.test_type == 'hard':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_hard.txt', str)
    elif args.test_type == 'extend':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_extended.txt', str)
    else:
        raise IOError('==> unknown test type.')

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, 250, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval',
                                                aggregation=args.aggregation_mode,
                                                loss=args.loss,
                                                vlad_clusters=args.vlad_cluster,
                                                ghost_clusters=args.ghost_cluster,
                                                bottleneck_dim=args.bottleneck_dim,
                                                net=args.net)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> guess the model name
        if args.aggregation_mode == 'avg':
            imag_model = 'bdim{}'.format(args.bottleneck_dim)
        elif args.aggregation_mode == 'vlad':
            imag_model = 'vlad{}_bdim{}'.format(args.vlad_cluster, args.bottleneck_dim)
        elif args.aggregation_mode == 'gvlad':
            imag_model = 'vlad{}_ghost{}_bdim{}_ohemlevel{}'.format(args.vlad_cluster, args.ghost_cluster,
                                                                    args.bottleneck_dim, args.ohem_level)
        else:
            raise IOError('==> unknown aggregation mode for model loading.')

        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            model_folder = args.resume.split(os.sep)[-2]
            real_model = '_'.join(model_folder[model_folder.find('lr'):].split('_')[1:])
            if imag_model == real_model:
                network_eval.load_weights(os.path.join(args.resume), by_name=True)
            else:
                print('==> real model : {}'.format(real_model))
                print('==> imagine model : {}'.format(imag_model))
                raise IOError('==> line 104, imagine model does not match real model, please check.')
            result_path = set_result_path(args)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))

    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
    for c, ID in enumerate(unique_list):
        if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
        specs = ut.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        v = network_eval.predict(specs)
        feats += [v]

    feats = np.array(feats)

    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v2 = feats[ind2, 0]

        scores += [np.sum(v1*v2)]
        labels += [verify_lb[c]]
        print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    np.save(os.path.join(result_path, 'prediction_scores.npy'), scores)
    np.save(os.path.join(result_path, 'groundtruth_labels.npy'), labels)

    eer, thresh = toolkits.calculate_eer(labels, scores)
    print('==> model : {}, EER: {}'.format(real_model, eer))


def set_result_path(args):
    model_path = args.resume
    exp_path = model_path.split(os.sep)
    result_path = os.path.join('../result', exp_path[2], exp_path[3])
    if not os.path.exists(result_path): os.makedirs(result_path)
    return result_path


if __name__ == "__main__":
    main()
