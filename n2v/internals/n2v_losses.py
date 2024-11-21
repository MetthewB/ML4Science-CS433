# @inproceedings{krull2019noise2void,
#   title={Noise2void-learning denoising from single noisy images},
#   author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={2129--2137},
#   year={2019}
# }

import tensorflow.keras.backend as K

import tensorflow as tf

def loss_mse():
    def n2v_mse(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        loss = tf.reduce_sum(K.square(target - y_pred*mask)) / tf.reduce_sum(mask)
        return loss

    return n2v_mse

def loss_mae():
    def n2v_abs(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        loss = tf.reduce_sum(K.abs(target - y_pred*mask)) / tf.reduce_sum(mask)
        return loss

    return n2v_abs