import collections
import json
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.python.training.saver import BaseSaverBuilder

import model

output_dir = 'results-baseline/'

REG_WEIGHT = 1e-6

INPUT_DIM = 28
INPUT_CHANNELS = 1
N_CLASSES = 10

COEFF_START = 0.0
n_adv_iters = 5000

#
orig_imgs = np.load(output_dir + 'mnist_orig_imgs.npy')
orig_labels = np.load(output_dir + 'mnist_orig_labels.npy')
orig_labels_idx = np.argmax(orig_labels, 1)
N_IMAGES = orig_imgs.shape[0]

# targets
target_labels_idx = np.random.randint(N_CLASSES - 1, size=orig_labels_idx.shape)
target_labels_idx += target_labels_idx >= orig_labels_idx
target_labels = np.eye(N_CLASSES)[target_labels_idx]
np.save(output_dir + 'mnist_target_labels.npy', target_labels)
if np.any(target_labels_idx == orig_labels_idx):
    print 'oops, target matches ground truth'

# Placeholders
X  = tf.placeholder(shape=(None, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS),
                    dtype=tf.float32, name='X')
y  = tf.placeholder(shape=(None, N_CLASSES), dtype=tf.float32, name='y')

x_star_tanh_init = np.arctanh((orig_imgs - 0.5) * 1.999)
x_star_tanh = tf.Variable(x_star_tanh_init, dtype=tf.float32)
x_star = tf.tanh(x_star_tanh) / 2. + 0.5

sess = tf.Session()

# Models
Model = collections.namedtuple('Model', ['logits', 'softmax'])
def unscope_name(n):
    slash_idx = n.index('/')
    return n[slash_idx + 1:]
class UnscopeBuilder(BaseSaverBuilder):
    def restore_op(self, filename_tensor, saveable, preferred_shard):
        return [io_ops.restore_v2(filename_tensor,
                                  [unscope_name(spec.name)],
                                  [spec.slice_spec],
                                  [spec.tensor.dtype])[0] for spec in saveable.specs]
def create_model(X, s):
    s_str = ''.join(str(cls) for cls in s)
    chkpt_dir = 'checkpoints-%s/' % s_str
    with tf.variable_scope('m%s' % s_str):
        global_before = tf.global_variables()
        logits, softmax = model.model(X, N_CLASSES, False, REG_WEIGHT)
        global_after = tf.global_variables()
        weights = list(set(global_after) - set(global_before))
    print 'restoring', [v.name for v in weights] # %%%
    # import pdb; pdb.set_trace() # %%%
    saver = tf.train.Saver(weights, builder=UnscopeBuilder())
    saver.restore(sess, chkpt_dir + 'mnist_weights.ckpt')
    return Model(logits, softmax)
generalist = create_model(x_star, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Adapted from Carlini's code
kappa = 6.9
coeff = tf.placeholder(shape=(N_IMAGES,), dtype=tf.float32)
l2dist = tf.reduce_sum(tf.square(x_star - X), [1, 2, 3])
logits = generalist.logits
target_logits = tf.reduce_sum(y * logits, 1)
top_other_logits = tf.reduce_max((1 - y) * logits - y * 10000, 1)
target_penalty = tf.maximum(-kappa, top_other_logits - target_logits)
confidence = tf.reduce_sum(generalist.softmax * y, 1)

loss = tf.add(target_penalty * coeff, l2dist)
train_adv_step = tf.train.AdamOptimizer(2e-2).minimize(loss, var_list=[x_star_tanh])
optimizer_variables = tf.global_variables()[-4:]
predictions = tf.argmax(logits, 1)

####################

# Initialize loss coefficients
coeff_block_log = np.tile([[COEFF_START], [float('nan')], [float('nan')]], (1, N_IMAGES))
coeff_curr_log = coeff_block_log[0]
coeff_high_log = coeff_block_log[1]
coeff_low_log = coeff_block_log[2]

# Collect best adversarial images
best_l2 = np.zeros((N_IMAGES,)) + float('nan')
best_coeff_log = np.zeros((N_IMAGES,)) + float('nan')
best_iter = np.zeros((N_IMAGES,)) + float('nan')
best_images = np.copy(orig_imgs)

conf_thresh = 0.998951

for _ in range(9):
    # Reset x_star_tanh and optimizer
    sess.run(tf.variables_initializer([x_star_tanh] + optimizer_variables))

    print coeff_curr_log # %%%

    curr_coeff = np.exp(coeff_curr_log)
    all_fail = np.ones((N_IMAGES,), dtype=np.bool)

    # Training loop
    for j in range(n_adv_iters):
        xst, preds, conf, l2d, _ = sess.run([x_star, predictions, confidence, l2dist, train_adv_step], feed_dict={
            X: orig_imgs,
            y: target_labels,
            coeff: curr_coeff,
        })
        adv_fail = np.logical_or(preds != target_labels_idx, conf < conf_thresh)
        all_fail = np.logical_and(all_fail, adv_fail)
        for i in range(N_IMAGES):
            if adv_fail[i]:
                continue
            if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
                best_l2[i] = l2d[i]
                best_coeff_log[i] = coeff_curr_log[i]
                best_iter[i] = j
                best_images[i] = xst[i]
        if j % 100 == 0:
            collected = N_IMAGES - np.count_nonzero(np.isnan(best_l2))
            print("Adv. training iter. {}/{} collected {} avg l2 {}".format(j, n_adv_iters, collected, np.average(np.sqrt(best_l2))))

    xst, preds, conf, l2d = sess.run([x_star, predictions, confidence, l2dist], feed_dict={
        X: orig_imgs,
        y: target_labels,
    })
    adv_fail = np.logical_or(preds != target_labels_idx, conf < conf_thresh)
    all_fail = np.logical_and(all_fail, adv_fail)
    for i in range(N_IMAGES):
        if adv_fail[i] or conf[i] < conf_thresh:
            continue
        if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
            best_l2[i] = l2d[i]
            best_coeff_log[i] = coeff_curr_log[i]
            best_iter[i] = n_adv_iters
            best_images[i] = xst[i]
    collected = N_IMAGES - np.count_nonzero(np.isnan(best_l2))
    print("Finished training {}/{} collected {} avg l2 {}".format(n_adv_iters, n_adv_iters, collected, np.average(np.sqrt(best_l2))))

    # Save generated examples and their coefficients
    np.save(output_dir + 'mnist_adv_imgs.npy', best_images)
    np.save(output_dir + 'mnist_adv_coeff_log.npy', best_coeff_log)

    # Update coeff
    for i, (fail, curr, high, low) in enumerate(zip(all_fail, coeff_curr_log, coeff_high_log, coeff_low_log)):
        if fail:
            # increase to allow more distortion
            coeff_low_log[i] = low = curr
            if math.isnan(high):
                coeff_curr_log[i] = curr + 2.3
            else:
                coeff_curr_log[i] = (high + low) / 2
        else:
            # decrease to penalize distortion
            coeff_high_log[i] = high = curr
            if math.isnan(low):
                coeff_curr_log[i] = curr - 0.69
            else:
                coeff_curr_log[i] = (high + low) / 2
    np.save(output_dir + 'mnist_coeff_log.npy', coeff_block_log)
