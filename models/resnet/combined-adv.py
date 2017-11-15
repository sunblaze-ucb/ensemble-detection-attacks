import math

import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.ops import variables

import squeeze
import median

DETECTOR_L1_THRESHOLD = 0.3076
COEFF_START = 0.0 # %%%
adv_lr = 9e-2
n_adv_iters = 5000

eval_dir = '/tmp/resnet_model/test'
log_root = '/tmp/resnet_model'
hps = resnet_model.HParams(batch_size=100,
                           num_classes=10,
                           min_lrn_rate=0.0001,
                           lrn_rate=0.1,
                           num_residual_units=5,
                           use_bottleneck=False,
                           weight_decay_rate=0.0002,
                           relu_leakiness=0.1,
                           optimizer='mom')

sess = tf.Session()

batch_images = np.load(eval_dir + '/batch_unquantized_images.npy')
batch_images_tanh = np.arctanh((batch_images - 0.5) / 0.501)
batch_labels = np.load(eval_dir + '/batch_labels.npy')

N0, H0, W0, C0 = batch_images.shape

X = tf.placeholder(shape=(N0, H0, W0, C0), dtype=tf.float32)
Y = tf.placeholder(shape=(N0, 10), dtype=tf.float32)

x_star_tanh = tf.Variable(batch_images_tanh, dtype=tf.float32)
x_star = tf.tanh(x_star_tanh) / 2. + 0.5
x_star_quantized = squeeze.reduce_precision_tf(x_star, 8)
x_star_smoothed = tf.reshape(median.median_filter(x_star, 2, 2), (N0, H0, W0, C0))
x_star_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_star)
x_star_quantized_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_star_quantized)
x_star_smoothed_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_star_smoothed)

# Build the model with our perturbed images as input
model = resnet_model.ResNet(hps, x_star_scaled, Y, 'eval')
model.build_graph()
model_variables = variables._all_saveable_objects()
model_variables.remove(x_star_tanh)
saver = tf.train.Saver(model_variables)
model_quantized = resnet_model.ResNet(hps, x_star_quantized_scaled, Y, 'eval')
model_quantized.build_graph(reuse=True)
model_smoothed = resnet_model.ResNet(hps, x_star_smoothed_scaled, Y, 'eval')
model_smoothed.build_graph(reuse=True)

# Adapted from Carlini's code
coeff = tf.placeholder(shape=(N0,), dtype=tf.float32)
l2dist = tf.reduce_sum(tf.square(x_star - X), [1, 2, 3])
ground_truth_logits = tf.reduce_sum(Y * model.logits, 1)
top_other_logits = tf.reduce_max((1 - Y) * model.logits - (Y * 10000), 1)
target_penalty = tf.maximum(0., ground_truth_logits - top_other_logits)
l1score_nq = tf.reduce_sum(tf.abs(model.predictions - model_quantized.predictions), 1)
l1score_ns = tf.reduce_sum(tf.abs(model.predictions - model_smoothed.predictions), 1)
l1score_qs = tf.reduce_sum(tf.abs(model_smoothed.predictions - model_quantized.predictions), 1)
detector_penalty_nq = tf.maximum(0., l1score_nq - DETECTOR_L1_THRESHOLD)
detector_penalty_ns = tf.maximum(0., l1score_ns - DETECTOR_L1_THRESHOLD)
detector_penalty_qs = tf.maximum(0., l1score_qs - DETECTOR_L1_THRESHOLD)
detector_penalty = detector_penalty_nq + detector_penalty_ns + detector_penalty_qs

loss = tf.add((target_penalty + detector_penalty) * coeff, l2dist)
train_adv_step = tf.train.AdamOptimizer(adv_lr).minimize(loss, var_list=[x_star_tanh])
optimizer_variables = tf.global_variables()[-4:]
predictions = tf.argmax(model.logits, 1)
correct_prediction = tf.equal(predictions, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Restore a checkpoint with the learned weights
try:
    ckpt_state = tf.train.get_checkpoint_state(log_root)
except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
    raise
if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('No model to eval yet at %s', log_root)
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
best_images = np.copy(batch_images)

for _ in range(9):
    # Reset x_star_tanh and optimizer
    sess.run(tf.variables_initializer([x_star_tanh] + optimizer_variables))
    tf.assert_variables_initialized()

    print coeff_curr_log # %%%
    curr_coeff = np.exp(coeff_curr_log)
    all_fail = np.ones((N0,), dtype=np.bool)

    # Training loop
    improve_count = 0
    for j in range(n_adv_iters):
        xst, adv_fail, l1o, l2d, _ = sess.run([x_star, correct_prediction, detector_penalty, l2dist, train_adv_step], feed_dict={
            X: batch_images,
            Y: batch_labels,
            coeff: curr_coeff,
        })
        all_fail = np.logical_and(all_fail, adv_fail)
        for i in range(N0):
            if adv_fail[i] or l1o[i] > 0:
                continue
            if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
                best_l2[i] = l2d[i]
                best_coeff_log[i] = coeff_curr_log[i]
                best_iter[i] = j
                best_images[i] = xst[i]
                improve_count += 1
        if j % 100 == 0:
            print("Adv. training iter. {}/{} improved {}".format(j, n_adv_iters, improve_count))
            improve_count = 0

    xst, adv_fail, l1o, l2d = sess.run([x_star, correct_prediction, detector_penalty, l2dist], feed_dict={
        X: batch_images,
        Y: batch_labels,
    })
    for i in range(N0):
        if adv_fail[i] or l1o[i] > 0:
            continue
        if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
            best_l2[i] = l2d[i]
            best_coeff_log[i] = coeff_curr_log[i]
            best_iter[i] = n_adv_iters
            best_images[i] = xst[i]
            improve_count += 1
    print("Finished training {}/{} improved {}".format(n_adv_iters, n_adv_iters, improve_count))

    # Save generated examples and their coefficients
    np.save(eval_dir + '/combined_adv_imgs.npy', best_images)
    np.save(eval_dir + '/combined_adv_coeff_log.npy', best_coeff_log)

    # Update coeff
    for i, (fail, curr, high, low) in enumerate(zip(adv_fail, coeff_curr_log, coeff_high_log, coeff_low_log)):
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
    np.save(eval_dir + '/combined_coeff_log.npy', coeff_block_log)
