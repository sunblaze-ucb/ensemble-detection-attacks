# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""adversarial examples generation
"""
import time
import six
import sys
import math

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.ops import variables

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'eval', 'eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

reg_weight = 0.1 # for L2
adv_lr = 9e-2
n_adv_iters = 5000
temp_init = 1.0

ANNEAL_RATE=0.00045
MIN_TEMP=0.1
COEFF_START = 0.0

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=True):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

def evaluate(hps):
  """Eval loop."""
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  batch_unquantized_images = np.load(FLAGS.eval_dir + '/batch_unquantized_images.npy')
  batch_unscaled_images = np.load(FLAGS.eval_dir + '/batch_unscaled_images.npy')
  batch_labels = np.load(FLAGS.eval_dir + '/batch_labels.npy')

  N0, H0, W0, C0 = batch_unscaled_images.shape

  # Initialize distribution weights
  x_star_dist_init = np.eye(8)[(batch_unscaled_images * 7).astype(np.int32)] * 0.992 + 0.001
  x_star_dist_shape = x_star_dist_init.shape
  x_star_dist = tf.Variable(x_star_dist_init, dtype=tf.float32)

  tau = tf.placeholder(tf.float32, name='temperature')

  # Sample from Gumbel softmax
  gs_x_star_probs = tf.reshape(gumbel_softmax(tf.reshape(x_star_dist, [-1, 8]), tau, hard=True), x_star_dist_shape)
  gs_x_star_values = np.arange(8) / 7.
  x_star = tf.reduce_sum(gs_x_star_probs * gs_x_star_values, 4)

  Xr = tf.placeholder(shape=(N0, H0, W0, C0), dtype=tf.float32, name='Xr')
  X = tf.placeholder(shape=(N0, H0, W0, C0), dtype=tf.float32, name='X')
  x_star_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_star) # Recenter and rescale to what the model expects
  coeff = tf.placeholder(shape=(N0,), dtype=tf.float32)
  Y = tf.placeholder(shape=(N0, hps.num_classes), dtype=tf.float32)

  # Build the model with our perturbed images as input
  model = resnet_model.ResNet(hps, x_star_scaled, Y, FLAGS.mode)
  model.build_graph()
  model_variables = variables._all_saveable_objects()
  model_variables.remove(x_star_dist)

  saver = tf.train.Saver(model_variables)
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  # Adapted from Carlini's code
  l2dist = tf.reduce_sum(tf.square(x_star - Xr), [1, 2, 3])
  ground_truth_logits = tf.reduce_sum(Y * model.logits, 1)
  top_other_logits = tf.reduce_max((1 - Y) * model.logits - (Y * 10000), 1)
  target_penalty = tf.maximum(0., ground_truth_logits - top_other_logits)

  loss = tf.add(target_penalty * coeff, l2dist)
  train_adv_step = tf.train.AdamOptimizer(adv_lr).minimize(loss, var_list=[x_star_dist])
  optimizer_variables = tf.global_variables()[-4:]
  predictions = tf.argmax(model.logits, 1)
  correct_prediction = tf.equal(predictions, tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Restore a checkpoint with the learned weights
  try:
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
  except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
    raise
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
    raise
  tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)

  # Initialize loss coefficients
  coeff_block_log = np.tile([[COEFF_START], [float('nan')], [float('nan')]], (1, N0))
  coeff_curr_log = coeff_block_log[0]
  coeff_high_log = coeff_block_log[1]
  coeff_low_log = coeff_block_log[2]

  # Collect best adversarial images
  best_l2 = np.zeros((N0,)) + float('nan')
  best_coeff_log = np.zeros((N0,)) + float('nan')
  best_iter = np.zeros((N0,)) + float('nan')
  best_images = np.copy(batch_unquantized_images)

  improve_count = 0
  for _ in range(9):
    # Reset x_star_dist and optimizer
    sess.run(tf.variables_initializer([x_star_dist] + optimizer_variables))
    tf.assert_variables_initialized()

    curr_temp = temp_init
    curr_coeff = np.exp(coeff_curr_log)
    print coeff_curr_log # %%%
    all_fail = np.ones((N0,), dtype=np.bool)

    # Training loop
    for j in range(n_adv_iters):
      xst, adv_fail, l2d, _ = sess.run([x_star, correct_prediction, l2dist, train_adv_step], feed_dict={
        Xr: batch_unquantized_images,
        X: batch_unscaled_images,
        Y: batch_labels,
        tau: curr_temp,
        coeff: curr_coeff,
      })
      all_fail = np.logical_and(all_fail, adv_fail)
      for i in range(N0):
        if adv_fail[i]:
          continue
        if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
          best_l2[i] = l2d[i]
          best_coeff_log[i] = coeff_curr_log[i]
          best_iter[i] = j
          best_images[i] = xst[i]
          improve_count += 1
      if j % 5 == 0:
        curr_temp = np.maximum(temp_init * np.exp(-ANNEAL_RATE * j), MIN_TEMP)
      if j % 100 == 0:
        print("Adv. training iter. {}/{} improved {}, temp {}".format(j, n_adv_iters, improve_count, curr_temp))
        improve_count = 0

    xst, adv_fail, l2d = sess.run([x_star, correct_prediction, l2dist], feed_dict={
      Xr: batch_unquantized_images,
      X: batch_unscaled_images,
      Y: batch_labels,
      tau: curr_temp,
    })
    all_fail = np.logical_and(all_fail, adv_fail)
    for i in range(N0):
      if adv_fail[i]:
        continue
      if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
        best_l2[i] = l2d[i]
        best_coeff_log[i] = coeff_curr_log[i]
        best_iter[i] = j
        best_images[i] = xst[i]
        improve_count += 1
    print("Finished training {}/{} improved {}".format(n_adv_iters, n_adv_iters, improve_count))

    # Save generated examples and their coefficients
    np.save(FLAGS.eval_dir + '/precision_adv_imgs.npy', best_images)
    np.save(FLAGS.eval_dir + '/precision_adv_coeff_log.npy', best_coeff_log)

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
    np.save(FLAGS.eval_dir + '/precision_coeff_log.npy', coeff_block_log)


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  with tf.device(dev):
    if FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
