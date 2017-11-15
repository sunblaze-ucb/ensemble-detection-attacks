import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.ops import variables

# import squeeze
from scipy import ndimage
import median

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
batch_labels = np.load(eval_dir + '/batch_labels.npy')
adv_images = np.load(eval_dir + '/smoothing_adv_imgs.npy')

N0, H0, W0, C0 = batch_images.shape

# Note: squeeze.median_filter_np operates on NCHW data; the below does the same for NHWC
# adv_images_median = squeeze.median_filter_np(adv_images, 2, 2)
adv_images_median = ndimage.filters.median_filter(adv_images, size=(1, 2, 2, 1), mode='reflect')
batch_images_median = ndimage.filters.median_filter(batch_images, size=(1, 2, 2, 1), mode='reflect')

# Adapted from Carlini's code
l2dist = np.sum(np.square(batch_images - adv_images), (1, 2, 3))
l2dist_median = np.sum(np.square(batch_images_median - adv_images_median), (1, 2, 3))

print 'L2 distance average', np.average(np.sqrt(l2dist))
print 'L2 distance average (after median)', np.average(np.sqrt(l2dist_median))

X = tf.placeholder(shape=(N0, H0, W0, C0), dtype=tf.float32)
Y = tf.placeholder(shape=(N0, 10), dtype=tf.float32)

# Build the model
# X_median = median.median_filter(X, 2, 2)
# X_median = tf.reshape(X_median, (N0, H0, W0, C0))
X_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), X)
model = resnet_model.ResNet(hps, X_scaled, Y, 'eval')
model.build_graph()
model_variables = variables._all_saveable_objects()
saver = tf.train.Saver(model_variables)

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

tf.assert_variables_initialized()

def test_batch(x, desc, probs_filename, preds_filename):
    probs, preds, acc = sess.run([model.predictions, predictions, accuracy], feed_dict={
        X: x,
        Y: batch_labels,
    })
    print desc, acc
    if probs_filename is not None:
        np.save(probs_filename, probs)
    if preds_filename is not None:
        np.save(preds_filename, preds)

test_batch(batch_images, 'acc orig no-filter',
           eval_dir + '/smoothing_orig_nofilter_probs.npy',
           eval_dir + '/smoothing_orig_nofilter_preds.npy')
test_batch(batch_images_median, 'acc orig median',
           eval_dir + '/smoothing_orig_median_probs.npy',
           eval_dir + '/smoothing_orig_median_preds.npy')
test_batch(adv_images, 'acc adv no-filter',
           eval_dir + '/smoothing_adv_nofilter_probs.npy',
           eval_dir + '/smoothing_adv_nofilter_preds.npy')
test_batch(adv_images_median, 'acc adv median',
           eval_dir + '/smoothing_adv_median_probs.npy',
           eval_dir + '/smoothing_adv_median_preds.npy')
