from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.median import *
from lib.bitdepth import *
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Cmd Line Example: python detection.py 1 2 2 100 5000 1 10

arg_names = ['bit_depth', 'med_filt_height', 'med_filt_width', 'n_adv_examples', 'n_adv_iters', 'restore_from_checkpoint', 'n_optimizers']
cmdline = zip(arg_names, sys.argv[1:])
argmap = {}
for name, val in cmdline:
    argmap[name] = int(val)

###################################
# Feature Squeezing Configuration #
###################################
med_filt_height = argmap.get('med_filt_height', 3)
med_filt_width  = argmap.get('med_filt_width', 3)
bit_depth       = argmap.get('bit_depth', 1)
n_adv_examples = argmap.get('n_adv_examples', 100)
n_adv_iters = argmap.get('n_adv_iters', 5000)
restore_from_checkpoint = argmap.get('restore_from_checkpoint', False)
n_optimizers = argmap.get('n_optimizers', 10)

adv_l2_reg_weight = 0.1
# adv_sm_reg_weight = 0.5
adv_lr = 2e-2
results_dir = 'results/detection_{}x{}_{}bits/'.format(med_filt_height, med_filt_width, bit_depth)
print_every = 100

######################
# Data Configuration #
######################
# chkpt_dir = 'weights/ckpts_detection_{}x{}_{}bits/'.format(med_filt_height, med_filt_width, bit_depth)
chkpt_dir = 'mnist-newmodel-weights/baseline/' # %%%
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

def mnist_net(X, y, lr, scope_prefix, reuse=False):
    if scope_prefix is None:
        scope_prefix = ''
    else:
        scope_prefix += '/'
    with tf.variable_scope('', reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(REG_WEIGHT), reuse=reuse):
            with slim.arg_scope([slim.conv2d], rate=(1,1)):
                drop1 = slim.dropout(X, keep_prob=0.8, is_training=tr_mode, scope=scope_prefix + 'dropout1')
                conv1 = slim.conv2d(drop1, 64 * 1, [8, 8], stride=(2,2), padding='SAME',  scope=scope_prefix + 'conv1')
                conv2 = slim.conv2d(conv1, 64 * 2, [6, 6], stride=(2,2), padding='VALID', scope=scope_prefix + 'conv2')
                conv3 = slim.conv2d(conv2, 64 * 2, [5, 5], stride=(1,1), padding='VALID', scope=scope_prefix + 'conv3')
                drop2 = slim.dropout(conv3, keep_prob=0.5, is_training=tr_mode, scope=scope_prefix + 'dropout2')
            flat = slim.flatten(drop2, scope=scope_prefix + 'flat')
            logits = slim.fully_connected(flat, N_CLASSES, activation_fn=None, scope=scope_prefix + 'fc')
            softmax = tf.nn.softmax(logits, name=scope_prefix + 'softmax')
    cross_entropy  = slim.losses.softmax_cross_entropy(softmax, y)
    regularization = tf.add_n(slim.losses.get_regularization_losses(scope=scope_prefix))
    loss = cross_entropy + regularization
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    train_step = None # %%% tf.train.AdamOptimizer(lr).minimize(loss)
    return logits, softmax, loss, correct_pred, accuracy, train_step

#########################
# Branch 1: No defenses #
#########################
logits, softmax, loss, correct_pred, accuracy, train_step = mnist_net(X, y, lr, None, reuse=False)

#################################
# Branch 2: Precision reduction #
#################################
X_pr = precision_filter(X, bit_depth)
logits_pr, softmax_pr, loss_pr, correct_pred_pr, accuracy_pr, train_step_pr = mnist_net(X_pr, y, lr, None, reuse=True)

##############################
# Branch 3: Median smoothing #
##############################
X_med = median_filter(X, med_filt_height, med_filt_width)
X_med = tf.reshape(X_med, (-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS))
logits_med, softmax_med, loss_med, correct_pred_med, accuracy_med, train_step_med = mnist_net(X_med, y, lr, None, reuse=True)

# Fire up the Tensorflow session!
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=8,
    gpu_options=gpu_options))
sess.run(init)
saver = tf.train.Saver()

x_test = np.reshape(data.test.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
test_dict = { X: x_test, y: data.test.labels, tr_mode: False }
fstr_test = "Test set performance: loss = {:.6f}, accuracy = {:.6f}\n"

def train_branch(loss, accuracy, train_step):    
    n_train = data.train.num_examples
    n_iters = int(np.ceil((1.0 * N_EPOCHS * n_train) / BATCH_SIZE))
    epoch = lambda i: int((i * BATCH_SIZE) / n_train) + 1
    anneal_rate = (-1.0 * np.log(LR_END / LR_START)) / float(n_iters)
    x_val = np.reshape(data.validation.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
    eval_dict = { X: x_val, y: data.validation.labels, tr_mode: False }
    best_loss, curr_lr, curr_epoch = 1.0e20, LR_START, 1
    fstr = """Iter {:05d} (epoch {:03d}) - validation loss = {:.6f}
         Validation accuracy.... = {:.6f}
         Checkpointed........... = {}
    """
    for i in range(n_iters):
        if i % 25 == 0 and anneal_rate > 0.0:
            if sess.run(loss, feed_dict=eval_dict) < best_loss:
                curr_lr = LR_START * np.exp(-1.0 * anneal_rate * i)
        batch = data.train.next_batch(BATCH_SIZE)
        x = np.reshape(batch[0], [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
        sess.run(train_step, feed_dict={ X: x, y: batch[1], lr: curr_lr, tr_mode: True })
        if curr_epoch != epoch(i + 1):
            val_loss, curr_epoch, checkpointed = sess.run(loss, feed_dict=eval_dict), epoch(i + 1), False
            if val_loss < best_loss:
                best_loss, checkpointed = val_loss, True
                saver.save(sess, chkpt_dir + 'mnist_weights.ckpt')
            print(fstr.format(i, epoch(i), val_loss, sess.run(accuracy, feed_dict=eval_dict), checkpointed))
    print("Optimization finished!")
    print("Restoring model with lowest validation loss...")
    saver.restore(sess, chkpt_dir + 'mnist_weights.ckpt')
    print(fstr_test.format(
        sess.run(loss,     feed_dict=test_dict),
        sess.run(accuracy, feed_dict=test_dict)))

if restore_from_checkpoint:
    print("Restoring model with lowest validation loss...")
    saver.restore(sess, chkpt_dir + 'mnist_weights.ckpt')
    # print("Branch 1: no defense")
    # print(fstr_test.format(
    #     sess.run(loss,     feed_dict=test_dict),
    #     sess.run(accuracy, feed_dict=test_dict)))
    # print("Branch 2: precision reduction")
    # print(fstr_test.format(
    #     sess.run(loss_pr,     feed_dict=test_dict),
    #     sess.run(accuracy_pr, feed_dict=test_dict)))
    # print("Branch 3: median filter")
    # print(fstr_test.format(
    #     sess.run(loss_med,     feed_dict=test_dict),
    #     sess.run(accuracy_med, feed_dict=test_dict)))
else:
    print("Training branch 1 (no defense)...")
    train_branch(loss, accuracy, train_step)
    print("Training branch 2 (precision reduction)...")
    train_branch(loss_pr, accuracy_pr, train_step_pr)
    print("Training branch 3 (median filter)...")
    train_branch(loss_med, accuracy_med, train_step_med)

examples_use = sess.run(correct_pred, feed_dict={
    X: x_test[:n_adv_examples],
    y: data.test.labels[:n_adv_examples],
    tr_mode: False,
})
print('using', np.count_nonzero(examples_use), 'classified correctly')

#################################
# Optimize Adversarial Examples #
#################################
x = np.reshape(data.test.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
ground_truth = data.test.labels
x = x[:n_adv_examples]
ground_truth = ground_truth[:n_adv_examples]

# TODO: rename delta*
delta_initial = np.arctanh((x - 0.5) / 0.501)

N0, H0, W0, C0 = x.shape

delta_shape = [N0, H0, W0, C0]
delta = tf.Variable(delta_initial, dtype=tf.float32)

x_star = tf.tanh(delta) / 2. + 0.5

# Branch 1
logits_adv, softmax_adv, loss_adv, correct_pred_adv, accuracy_adv, train_step_adv = mnist_net(x_star, y, lr, None, reuse=True)
# Branch 2
X_pr = precision_filter(x_star, bit_depth)
logits_pr_adv, softmax_pr_adv, loss_pr_adv, correct_pred_pr_adv, accuracy_pr_adv, train_step_pr_adv = mnist_net(X_pr, y, lr, None, reuse=True)
# Branch 3
X_med = median_filter(x_star, med_filt_height, med_filt_width)
X_med = tf.reshape(X_med, (-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS))
logits_med_adv, softmax_med_adv, loss_med_adv, correct_pred_med_adv, accuracy_med_adv, train_step_med_adv = mnist_net(X_med, y, lr, None, reuse=True)



#################
# LOSS FUNCTION #
# (not working) #
#################
# Softmax L1 difference
L1_THRESHOLD = 0.3076
l1_pr  = tf.reduce_sum(tf.abs(softmax_adv - softmax_pr_adv), [1])
l1_med = tf.reduce_sum(tf.abs(softmax_adv - softmax_med_adv), [1])
l1_pr_med = tf.reduce_sum(tf.abs(softmax_pr_adv - softmax_med_adv), [1])
l1_score = tf.maximum(tf.maximum(l1_pr, l1_med), l1_pr_med)

# L2 attack, adapted from Carlini's code
l2dist = tf.reduce_sum(tf.square(x_star - X), [1, 2, 3])
l2_penalty = l2dist * adv_l2_reg_weight
kappa = 2.

target_logits = tf.reduce_sum(y * logits_adv, 1)
top_other_logits = tf.reduce_max((1 - y) * logits_adv - (y * 10000), 1)
target_penalty = tf.maximum(-kappa, top_other_logits - target_logits)

target_logits_med = tf.reduce_sum(y * logits_med_adv, 1)
top_other_logits_med = tf.reduce_max((1 - y) * logits_med_adv - (y * 10000), 1)
target_penalty_med = tf.maximum(-kappa, top_other_logits_med - target_logits_med)

target_logits_pr = tf.reduce_sum(y * logits_pr_adv, 1)
top_other_logits_pr = tf.reduce_max((1 - y) * logits_pr_adv - (y * 10000), 1)
target_penalty_pr = tf.maximum(-kappa, top_other_logits_pr - target_logits_pr)

# Objective function
adv_train_loss = target_penalty + target_penalty_med + l2_penalty

# Random optimizer initializations
train_adv_step = tf.train.AdamOptimizer(adv_lr).minimize(adv_train_loss, var_list=[delta])
train_opt_vars = tf.global_variables()[-4:]
for v in train_opt_vars:
    print('opt var', v.name) # %%%

# Veriable initialization
uninitialized = []
for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
    if not sess.run(tf.is_variable_initialized(v)):
        uninitialized.append(v)
        print('uninit', v.name) # %%%
sess.run(tf.variables_initializer(uninitialized))
sess.run(delta.assign(delta_initial))

np.save(results_dir + 'original_images.npy', x)
np.save(results_dir + 'ground_truth_labels.npy', ground_truth)
fstr = """Iter {:04d}/{:04d} (optimizer {:02d}/{:02d})
         Model accuracy......... = {:.4f}
         Average best penalty... = {:.4f}
         Average suspicion...... = {:.4f}
    """

# targets
ground_truth_idx = np.argmax(ground_truth, 1)
target_labels_idx = np.random.randint(N_CLASSES - 1, size=ground_truth_idx.shape)
target_labels_idx += target_labels_idx >= ground_truth_idx
target_labels = np.float32(np.eye(N_CLASSES)[target_labels_idx])
np.save(results_dir + 'adversarial_targets.npy', target_labels)

# Adversarial Training Loop
if os.path.exists(results_dir + 'l2.npy'):
    best_adv_imgs = np.load(results_dir + 'adversarial_images.npy')
    best_l2 = np.load(results_dir + 'l2.npy')
    best_l1s = np.load(results_dir + 'l1s.npy')
else:
    best_adv_imgs = np.copy(x)
    best_l2  = -1.0 * np.ones(n_adv_examples)
    best_l1s = -1.0 * np.ones(n_adv_examples)

np.set_printoptions(suppress=True)
for optimizer_id in range(n_optimizers):
    print("Optimizer {}/{}...".format(optimizer_id + 1, n_optimizers))
    # sess.run(delta.assign(delta_initial))
    sess.run(delta.assign(np.float32(np.random.normal(delta_initial, 0.5))))
    for j in range(n_adv_iters):
        sess.run(train_adv_step, feed_dict={ 
            X: x,
            y: target_labels,
            tr_mode: False
        })
        batch_l2, batch_tp, batch_l1s, batch_tppr, batch_tpmed, batch_xs = sess.run([l2dist, target_penalty, l1_score, target_penalty_pr, target_penalty_med, x_star], 
            feed_dict={ X: x, y: target_labels, tr_mode: False })
        for k in range(n_adv_examples):
            if (batch_tp[k] <= 0) and (batch_l1s[k] <= L1_THRESHOLD) and (best_l2[k] == -1.0 or batch_l2[k] < best_l2[k]):
                # print('ok', j, k); import code; code.interact(local=dict(globals(), **locals())); exit(1) # %%%
                best_l2[k] = batch_l2[k]
                best_l1s[k] = batch_l1s[k]
                best_adv_imgs[k, :, :, :] = batch_xs[k, :, :, :]
        if (j % print_every == 0 or j == n_adv_iters - 1):
            model_acc = (1.0 * np.sum(best_l2 == -1.0)) / (1.0 * n_adv_examples)
            if len(best_l2[best_l2 >= 0]) == 0:
                avg_penalty = -1.0
                avg_suspicion = -1.0
            else:
                avg_penalty = np.mean(np.sqrt(best_l2[best_l2 >= 0]))
                avg_suspicion = np.mean(best_l1s[best_l2 >= 0])
            np.save(results_dir + 'adversarial_images.npy', best_adv_imgs)
            np.save(results_dir + 'l2.npy', best_l2)
            np.save(results_dir + 'l1s.npy', best_l1s)
            print(fstr.format(j, n_adv_iters, optimizer_id + 1, n_optimizers, model_acc, avg_penalty, avg_suspicion))
            # print(best_l1s[best_l2 >= 0]) # %%%
            print(batch_tp[best_l2 == -1]) # %%%
            print(batch_tppr[best_l2 == -1]) # %%%
            print(batch_tpmed[best_l2 == -1]) # %%%

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
logits, softmax, loss, correct_pred, accuracy, train_step = mnist_net(X, y, lr, None, reuse=False)
X_pr = precision_filter(X, bit_depth)
logits_pr, softmax_pr, loss_pr, correct_pred_pr, accuracy_pr, train_step_pr = mnist_net(X_pr, y, lr, None, reuse=True)
X_med = median_filter(X, med_filt_height, med_filt_width)
X_med = tf.reshape(X_med, (-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS))
logits_med, softmax_med, loss_med, correct_pred_med, accuracy_med, train_step_med = mnist_net(X_med, y, lr, None, reuse=True)
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

l1_pr  = tf.reduce_sum(tf.abs(softmax - softmax_pr), [1])
l1_med = tf.reduce_sum(tf.abs(softmax - softmax_med), [1])
l1_prmed = tf.reduce_sum(tf.abs(softmax_pr - softmax_med), [1])
nonadv_l1pr  = sess.run(l1_pr, feed_dict={X: x, tr_mode: False})
nonadv_l1med = sess.run(l1_med, feed_dict={X: x, tr_mode: False})
nonadv_l1prmed = sess.run(l1_prmed, feed_dict={X: x, tr_mode: False})
adv_l1pr     = sess.run(l1_pr, feed_dict={X: x_adv, tr_mode: False})
adv_l1med    = sess.run(l1_med, feed_dict={X: x_adv, tr_mode: False})
adv_l1prmed    = sess.run(l1_prmed, feed_dict={X: x_adv, tr_mode: False})
sm_scores = np.zeros([x.shape[0], 6])
sm_scores[:, 0] = nonadv_l1pr
sm_scores[:, 1] = nonadv_l1med
sm_scores[:, 2] = nonadv_l1prmed
sm_scores[:, 3] = adv_l1pr
sm_scores[:, 4] = adv_l1med
sm_scores[:, 5] = adv_l1prmed
np.save(results_dir + 'softmax_scores.npy', sm_scores)

acc = sess.run(accuracy, feed_dict={ X: x, y: ground_truth, tr_mode: False })
print("Non-adversarial accuracy (wrt ground truth) = " + "{:.6f}".format(acc))
acc = sess.run(accuracy, feed_dict={ X: x_adv, y: ground_truth, tr_mode: False })
print("Adversarial accuracy (wrt ground truth) = " + "{:.6f}".format(acc))
acc = sess.run(accuracy, feed_dict={ X: x_adv, y: target_labels, tr_mode: False })
print("Adversarial accuracy (wrt target labels) = " + "{:.6f}".format(acc))

cp = sess.run(correct_pred, feed_dict={X: x_adv, y: ground_truth, tr_mode: False})
l2_all = np.sqrt(np.sum(np.square(x_adv - x), axis=(1, 2, 3)))
l2_mean = np.mean(l2_all[cp == 0])
print("Average L2 distance (successful adv. examples only): {:.6f}".format(l2_mean))
