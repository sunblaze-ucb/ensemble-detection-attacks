import tensorflow as tf
import tensorflow.contrib.slim as slim

# Model - CleverHans Keras MNIST model reimplementation w/ dropout
def model_(X, n_classes, tr_mode, reg_weight):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(reg_weight)):
        with slim.arg_scope([slim.conv2d], rate=(1,1)):
            drop1 = slim.dropout(X, keep_prob=0.8, is_training=tr_mode, scope='dropout1')
            conv1 = slim.conv2d(drop1, 64 * 1, [8, 8], stride=(2,2), padding='SAME',  scope='conv1')
            conv2 = slim.conv2d(conv1, 64 * 2, [6, 6], stride=(2,2), padding='VALID', scope='conv2')
            conv3 = slim.conv2d(conv2, 64 * 2, [5, 5], stride=(1,1), padding='VALID', scope='conv3')
            drop2 = slim.dropout(conv3, keep_prob=0.5, is_training=tr_mode, scope='dropout2')
        flat    = slim.flatten(drop2, scope='flat')
        logits  = slim.fully_connected(flat, n_classes, activation_fn=None, scope='fc')
        softmax = tf.nn.softmax(logits, name='softmax')
    return logits, softmax

# model from Appenix A.1:
# a CNN with three convolutional layers and one fully connected layer,
# where each convolutional layer is interlaced with ReLU, local
# contrast normalization, and a pooling layer. For regularization,
# dropout is used at the last layer, i.e., fully-connected layer, with
# p = 0.5.
def model(X, n_classes, tr_mode, reg_weight):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(reg_weight)):
        conv1 = slim.conv2d(X, 64 * 1, [8, 8], stride=(2,2), padding='SAME',  scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = slim.conv2d(conv1, 64 * 2, [6, 6], stride=(2,2), padding='VALID', scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
        conv3 = slim.conv2d(conv2, 64 * 2, [5, 5], stride=(1,1), padding='VALID', scope='conv3')
        drop = slim.dropout(conv3, keep_prob=0.5, is_training=tr_mode, scope='dropout')
        flat    = slim.flatten(drop, scope='flat')
        logits  = slim.fully_connected(flat, n_classes, activation_fn=None, scope='fc')
        softmax = tf.nn.softmax(logits, name='softmax')
    return logits, softmax
