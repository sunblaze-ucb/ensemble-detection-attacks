from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.bitdepth import *
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Cmd Line Example: python quantization_no_gs.py 1 100 5000 0

arg_names = ['bit_depth', 'n_adv_examples', 'n_adv_iters', 'restore_from_checkpoint']
cmdline = zip(arg_names, sys.argv[1:])
argmap = {}
for name, val in cmdline:
    argmap[name] = int(val)

###################################
# Feature Squeezing Configuration #
###################################
bit_depth = argmap.get('bit_depth', 1)
n_adv_examples = argmap.get('n_adv_examples', 100)
n_adv_iters = argmap.get('n_adv_iters', 5000)
restore_from_checkpoint = argmap.get('restore_from_checkpoint', False)

adv_l2_reg_weight = 0.1
adv_lr = 2e-2
n_optimizers = 10
results_dir = 'results/quantization_no_gs_{}bits/'.format(bit_depth)
print_every = 500

######################
# Data Configuration #
######################
chkpt_dir = 'weights/ckpts_quantization_{}bits/'.format(bit_depth)
data_dir = 'mnist_data'

# Load in data
data = input_data.read_data_sets(data_dir, one_hot=True)
print(" ") # Skip a line

def init_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

init_dir(chkpt_dir)
init_dir(results_dir)

##############################
# Target Model Configuration #
##############################
REG_WEIGHT = 1e-6
LR_START   = 9e-4
LR_END     = 7e-5

N_EPOCHS = 30
BATCH_SIZE = 128

INPUT_DIM = 28
INPUT_CHANNELS = 1
N_CLASSES = 10

####################
# Tensorflow Model #
####################

# Placeholders
X  = tf.placeholder(shape=(None, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS), 
                    dtype=tf.float32, name='X')
y  = tf.placeholder(shape=(None, N_CLASSES), dtype=tf.float32, name='y')
lr = tf.placeholder(tf.float32)
tr_mode = tf.placeholder(tf.bool)

# Preprocessing (quantization)
X_pp = precision_filter(X, bit_depth)

# Model - CleverHans Keras MNIST model reimplementation w/ dropout
with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(REG_WEIGHT)):
    with slim.arg_scope([slim.conv2d], rate=(1,1)):
        drop1 = slim.dropout(X_pp, keep_prob=0.8, is_training=tr_mode, scope='dropout1')
        conv1 = slim.conv2d(drop1, 64 * 1, [8, 8], stride=(2,2), padding='SAME',  scope='conv1')
        conv2 = slim.conv2d(conv1, 64 * 2, [6, 6], stride=(2,2), padding='VALID', scope='conv2')
        conv3 = slim.conv2d(conv2, 64 * 2, [5, 5], stride=(1,1), padding='VALID', scope='conv3')
        drop2 = slim.dropout(conv3, keep_prob=0.5, is_training=tr_mode, scope='dropout2')
    flat    = slim.flatten(drop2, scope='flat')
    logits  = slim.fully_connected(flat, N_CLASSES, activation_fn=None, scope='fc')
    softmax = tf.nn.softmax(logits, name='softmax')

# Loss function
cross_entropy  = slim.losses.softmax_cross_entropy(softmax, y)
regularization = tf.add_n(slim.losses.get_regularization_losses())

# loss = cross_entropy # Disable regularization (following CleverHans model)
loss = cross_entropy + regularization

# Evaluation metrics
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Optimize
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Fire up the Tensorflow session!
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=8,
    gpu_options=gpu_options))
sess.run(init)
saver = tf.train.Saver()

# Set up test set
x_test = np.reshape(data.test.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
test_dict = { X: x_test, y: data.test.labels, tr_mode: False }
fstr_test = "Test set performance: loss = {:.6f}, accuracy = {:.6f}\n"

if restore_from_checkpoint:
    print("Restoring model with lowest validation loss...")
    saver.restore(sess, chkpt_dir + 'mnist_weights.ckpt')
    print(fstr_test.format(
        sess.run(loss,     feed_dict=test_dict),
        sess.run(accuracy, feed_dict=test_dict)))
else:
    # Compute training parameters
    n_train = data.train.num_examples
    n_iters = int(np.ceil((1.0 * N_EPOCHS * n_train) / BATCH_SIZE))
    epoch = lambda i: int((i * BATCH_SIZE) / n_train) + 1
    anneal_rate = (-1.0 * np.log(LR_END / LR_START)) / float(n_iters)

    # Set up validation set
    x_val = np.reshape(data.validation.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
    eval_dict = { X: x_val, y: data.validation.labels, tr_mode: False }

    #################
    # Training loop #
    #################
    best_loss, curr_lr, curr_epoch = 1.0e20, LR_START, 1
    fstr = """Iter {:05d} (epoch {:03d}) - validation loss = {:.6f}
         Validation accuracy.... = {:.6f}
         Checkpointed........... = {}
    """

    for i in range(n_iters):
        # Anneal learning rate
        if i % 25 == 0 and anneal_rate > 0.0:
            if sess.run(loss, feed_dict=eval_dict) < best_loss:
                curr_lr = LR_START * np.exp(-1.0 * anneal_rate * i)

        # Train step
        batch = data.train.next_batch(BATCH_SIZE)
        x = np.reshape(batch[0], [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
        sess.run(train_step, feed_dict={ X: x, y: batch[1], lr: curr_lr, tr_mode: True })
        
        # Log and checkpoint @ end of each epoch
        if curr_epoch != epoch(i + 1):
            val_loss, curr_epoch, checkpointed = sess.run(loss, feed_dict=eval_dict), epoch(i + 1), False
            if val_loss < best_loss:
                best_loss, checkpointed = val_loss, True
                saver.save(sess, chkpt_dir + 'mnist_weights.ckpt')
            print(fstr.format(i, epoch(i), val_loss, sess.run(accuracy, feed_dict=eval_dict), checkpointed))

    # Finish training
    print("Optimization finished!")
    saver.save(sess, chkpt_dir + 'max_iters.ckpt') # Save final weights

    # Test set evaluation
    print(fstr_test.format(
        sess.run(loss,     feed_dict=test_dict),
        sess.run(accuracy, feed_dict=test_dict)))

    # Restore -best- model
    print("Restoring model with lowest validation loss...")
    saver.restore(sess, chkpt_dir + 'mnist_weights.ckpt')
    print(fstr_test.format(
        sess.run(loss,     feed_dict=test_dict),
        sess.run(accuracy, feed_dict=test_dict)))

orig_correct_pred = correct_pred

#################################
# Optimize Adversarial Examples #
#################################
x = np.reshape(data.test.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
ground_truth = data.test.labels
x = x[:n_adv_examples]
ground_truth = ground_truth[:n_adv_examples]

N0, H0, W0, C0 = x.shape

delta_shape = [N0, H0, W0, C0]
delta_initial = tf.constant(np.zeros_like(x), dtype=tf.float32)
delta = tf.Variable(delta_initial, dtype=tf.float32)

x_star = tf.add(X, delta)
x_star = tf.minimum(x_star, 1.0)
x_star = tf.maximum(x_star, 0.0)

with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(REG_WEIGHT), reuse=True):
    with slim.arg_scope([slim.conv2d], rate=(1,1)):
        drop1 = slim.dropout(x_star, keep_prob=0.8, is_training=tr_mode, scope='dropout1')
        conv1 = slim.conv2d(drop1, 64 * 1, [8, 8], stride=(2,2), padding='SAME',  scope='conv1')
        conv2 = slim.conv2d(conv1, 64 * 2, [6, 6], stride=(2,2), padding='VALID', scope='conv2')
        conv3 = slim.conv2d(conv2, 64 * 2, [5, 5], stride=(1,1), padding='VALID', scope='conv3')
        drop2 = slim.dropout(conv3, keep_prob=0.5, is_training=tr_mode, scope='dropout2')
    flat    = slim.flatten(drop2, scope='flat')
    logits  = slim.fully_connected(flat, N_CLASSES, activation_fn=None, scope='fc')
    softmax = tf.nn.softmax(logits, name='softmax')

# L2 attack, adapted from Carlini's code
l2dist = tf.reduce_sum(tf.square(x_star - X), [1, 2, 3])
l2_penalty = l2dist * adv_l2_reg_weight / N0
ground_truth_logits = tf.reduce_sum(y * logits, 1)
top_other_logits = tf.reduce_max((1 - y) * logits - (y * 10000), 1)
target_penalty = tf.maximum(0., ground_truth_logits - top_other_logits)
loss = tf.add(target_penalty, l2_penalty)

train_adv_steps = [tf.train.AdamOptimizer(adv_lr).minimize(loss, var_list=[delta])]
for _ in range(n_optimizers - 1):
    noisy_lr = adv_lr + np.random.uniform(low=-adv_lr, high=adv_lr)
    train_adv_steps.append(tf.train.AdamOptimizer(noisy_lr).minimize(loss, var_list=[delta]))

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Veriable initialization
uninitialized = []
for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
    if not sess.run(tf.is_variable_initialized(v)):
        uninitialized.append(v)
sess.run(tf.variables_initializer(uninitialized))
sess.run(delta.assign(delta_initial))

np.save(results_dir + 'original_images.npy', x)
np.save(results_dir + 'ground_truth_labels.npy', ground_truth)
fstr = """Iter {:04d}/{:04d} (optimizer {:02d}/{:02d})
         Model accuracy......... = {:.4f}
         Average best L2........ = {:.4f}
    """

# Adversarial Training Loop
best_adv_imgs = np.copy(x)
best_l2       = -1.0 * np.ones(n_adv_examples)

for optimizer_id, train_adv_step in enumerate(train_adv_steps):
    print("Optimizer {}/{}...".format(optimizer_id + 1, n_optimizers))
    sess.run(delta.assign(delta_initial))
    for j in range(n_adv_iters):
        # Train on unquantized samples...
        sess.run(train_adv_step, feed_dict={ 
            X: x, 
            y: ground_truth, 
            tr_mode: False
        })
        batch_l2, batch_xs = sess.run([l2dist, x_star], 
            feed_dict={ X: x, y: ground_truth, tr_mode: False })
        # But evaluate on quantized network
        batch_acc = sess.run(orig_correct_pred, feed_dict={ X: batch_xs, y: ground_truth, tr_mode: False })
        for k in range(n_adv_examples):
            if (batch_acc[k] == 0) and (batch_l2[k] < best_l2[k] or best_l2[k] == -1.0):
                best_l2[k] = batch_l2[k]
                best_adv_imgs[k, :, :, :] = batch_xs[k, :, :, :]
        if (j % print_every == 0 or j == n_adv_iters - 1):
            model_acc = (1.0 * np.sum(best_l2 == -1.0)) / (1.0 * n_adv_examples)
            if len(best_l2[best_l2 >= 0]) == 0:
                avg_l2 = -1.0
            else:
                avg_l2 = np.mean(np.sqrt(best_l2[best_l2 >= 0]))
            np.save(results_dir + 'adversarial_images.npy', best_adv_imgs)
            print(fstr.format(j, n_adv_iters, optimizer_id + 1, n_optimizers, model_acc, avg_l2))

print("Finished training")
sess.close()
tf.reset_default_graph()

####################
# Final Evaluation #
####################
X  = tf.placeholder(shape=(None, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS), 
                    dtype=tf.float32, name='X')
y  = tf.placeholder(shape=(None, N_CLASSES), dtype=tf.float32, name='y')
lr = tf.placeholder(tf.float32)
tr_mode = tf.placeholder(tf.bool)
X_pp = precision_filter(X, bit_depth)
with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(REG_WEIGHT)):
    with slim.arg_scope([slim.conv2d], rate=(1,1)):
        drop1 = slim.dropout(X_pp, keep_prob=0.8, is_training=tr_mode, scope='dropout1')
        conv1 = slim.conv2d(drop1, 64 * 1, [8, 8], stride=(2,2), padding='SAME',  scope='conv1')
        conv2 = slim.conv2d(conv1, 64 * 2, [6, 6], stride=(2,2), padding='VALID', scope='conv2')
        conv3 = slim.conv2d(conv2, 64 * 2, [5, 5], stride=(1,1), padding='VALID', scope='conv3')
        drop2 = slim.dropout(conv3, keep_prob=0.5, is_training=tr_mode, scope='dropout2')
    flat    = slim.flatten(drop2, scope='flat')
    logits  = slim.fully_connected(flat, N_CLASSES, activation_fn=None, scope='fc')
    softmax = tf.nn.softmax(logits, name='softmax')
cross_entropy  = slim.losses.softmax_cross_entropy(softmax, y)
regularization = tf.add_n(slim.losses.get_regularization_losses())
loss = cross_entropy + regularization
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=8,
    gpu_options=gpu_options))
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, chkpt_dir + 'mnist_weights.ckpt')
print("Model restored!")

x = np.load(results_dir + 'original_images.npy')
x_adv = np.load(results_dir + 'adversarial_images.npy')
ground_truth = np.load(results_dir + 'ground_truth_labels.npy')

nonadv_preds = sess.run(tf.argmax(logits, 1), feed_dict={X: x, tr_mode: False})
adv_preds    = sess.run(tf.argmax(logits, 1), feed_dict={X: x_adv, tr_mode: False})
np.save(results_dir + 'nonadversarial_predictions.npy', nonadv_preds)
np.save(results_dir + 'adversarial_predictions.npy', adv_preds)

acc = sess.run(accuracy, feed_dict={ X: x, y: ground_truth, tr_mode: False })
print("Non-adversarial accuracy (wrt ground truth) = " + "{:.6f}".format(acc))
acc = sess.run(accuracy, feed_dict={ X: x_adv, y: ground_truth, tr_mode: False })
print("Adversarial accuracy (wrt ground truth) = " + "{:.6f}".format(acc))

cp = sess.run(correct_pred, feed_dict={X: x_adv, y: ground_truth, tr_mode: False})
l2_all = np.sqrt(np.sum(np.square(x_adv - x), axis=(1, 2, 3)))
l2_mean = np.mean(l2_all[cp == 0])
print("Average L2 distance (successful adv. examples only): {:.6f}".format(l2_mean))
