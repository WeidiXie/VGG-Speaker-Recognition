# System
import tensorflow as tf
import keras.backend as K
from keras.engine import Layer

class L2_Norm(Layer):
    def __init__(self, **kwargs):
        super(L2_Norm, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, x):
        return K.l2_normalize(x, -1)


class Vlad_pooling(Layer):
    '''
    This layer follows the NetVlad.
    '''
    def __init__(self, k_centers, **kwargs):
        self.k_centers = k_centers
        super(Vlad_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers*input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        # Note here, in this particular speaker recognition example, normaly W = 1,
        feat, cluster_score = x
        num_features = feat.shape[-1]
        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims = True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)    # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)    # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster    # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)     # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])
        # Cluster intra-normalization.
        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, self.k_centers*num_features])
        return outputs


class Ghost_vlad_pooling(Layer):
    '''
    This layer follows the NetVlad.
    '''
    def __init__(self, k_centers, g_centers, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        super(Ghost_vlad_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[(self.k_centers + self.g_centers), input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        #return [(input_shape[0][0], input_shape[0][2], self.k_centers+self.g_centers),
        #        (input_shape[0][0], self.k_centers*input_shape[0][-1])]
        return input_shape[0][0], self.k_centers*input_shape[0][-1]

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        # Note here, in this particular speaker recognition example, normaly W = 1,
        feat, cluster_score = x
        H = feat.shape[2]
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        anneal_cluster_score = (cluster_score - max_cluster_score)/self.annealing
        exp_cluster_score = K.exp(anneal_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims=True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)    # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)    # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster    # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)     # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        # Cluster intra-normalization.
        cluster_res_no_g = cluster_res[:, :self.k_centers, :]
        cluster_l2 = K.l2_normalize(cluster_res_no_g, -1)

        outputs = K.reshape(cluster_l2, [-1, self.k_centers*num_features])
        #A_r = K.reshape(A, [-1, H, self.k_centers+self.g_centers])
        return outputs

