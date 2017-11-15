import collections
import json
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.python.training.saver import BaseSaverBuilder

import model

output_dir = 'results/'

REG_WEIGHT = 1e-6

INPUT_DIM = 28
INPUT_CHANNELS = 1
N_CLASSES = 10
N_MODELS = 2 * N_CLASSES + 1

# get those U sets
with open('sets-mnist.json', 'r') as f:
    u = json.load(f)
u_bool = np.zeros((N_CLASSES, N_MODELS), dtype=np.bool)
for i, s in enumerate(u):
    u_bool[s, i] = True

#
orig_imgs = np.load(output_dir + 'mnist_orig_imgs.npy')
orig_labels = np.load(output_dir + 'mnist_orig_labels.npy')
orig_labels_idx = np.argmax(orig_labels, 1)
orig_mask = u_bool[orig_labels_idx].transpose([1, 0])
N_IMAGES = orig_imgs.shape[0]
xst = np.load(output_dir + 'mnist_adv_imgs.npy')

# targets
target_labels = np.load(output_dir + 'mnist_target_labels.npy')
target_labels_idx = np.argmax(target_labels, 1)
mask = u_bool[target_labels_idx].transpose([1, 0])
if np.any(target_labels_idx == orig_labels_idx):
    print 'oops, target matches ground truth'

# Placeholders
X  = tf.placeholder(shape=(None, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS),
                    dtype=tf.float32, name='X')
y  = tf.placeholder(shape=(None, N_CLASSES), dtype=tf.float32, name='y')
x_star = tf.placeholder(shape=(None, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS), dtype=tf.float32, name='X')

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
models = [create_model(x_star, s) for s in u]

# Adapted from Carlini's code
l2dist = tf.reduce_sum(tf.square(x_star - X), [1, 2, 3])
logits = tf.stack([m.logits for m in models])

predictions = tf.argmax(logits, 2)
probs = tf.stack([m.softmax for m in models])
confidence = tf.reduce_max(probs, 2)

####################
np.set_printoptions(suppress=True)

# test original images too
preds, l2d, pr, conf = sess.run([predictions, l2dist, probs, confidence], feed_dict={
    X: orig_imgs,
    x_star: orig_imgs,
})

gen_fail = preds[-1] != orig_labels_idx
gt_predict = 1. - np.average(gen_fail.astype(np.float32))
print 'ground truth predicted by generalist', gt_predict
naive_mask = np.logical_not(gen_fail)
naive_confidence = np.average(conf[-1, naive_mask])
print 'generalist ones\' average confidence', naive_confidence
print 'failed on', np.where(gen_fail)
print ''

# for i in range(N_IMAGES):
#     target_label = target_labels_idx[i]
#     active = u_bool[target_label]
#     print target_label, preds[active, i], preds[:, i]
orig_fail = np.any(np.logical_and(orig_mask, preds != orig_labels_idx[None, :]), 0)
gt_predict = 1. - np.average(orig_fail.astype(np.float32))
print 'ground truth unanimously predicted', gt_predict
unanimous_mask = np.logical_and(orig_mask, np.logical_not(orig_fail))
unanimous_confidence = np.average(conf[unanimous_mask])
print 'unanimous ones\' average confidence', unanimous_confidence
print 'average l2 dist (sqrt)', np.average(np.sqrt(l2d))
print 'failed on', np.where(orig_fail)
print ''

print 'ground truth', orig_labels_idx[orig_fail]
xx = preds[:, orig_fail]
print 'preds', preds[:, orig_fail]
print 'probs', np.average(pr[:, orig_fail, :], 0)
print ''

# try it out
preds, l2d, conf = sess.run([predictions, l2dist, confidence], feed_dict={
    X: orig_imgs,
    x_star: xst,
})

gen_adv_fail = preds[-1] != target_labels_idx
target_predict = 1. - np.average(gen_adv_fail.astype(np.float32))
print 'target predicted by generalist', target_predict
naive_mask = np.logical_not(gen_adv_fail)
naive_confidence = np.average(conf[-1, naive_mask])
print 'generalist ones\' average confidence', naive_confidence
print 'average l2 dist (sqrt)', np.average(np.sqrt(l2d[np.logical_not(gen_adv_fail)]))
print 'failed on', np.where(gen_adv_fail)
print ''

preds, l2d, conf = np.asarray(sess.run([predictions, l2dist, confidence], feed_dict={
    X: orig_imgs,
    x_star: xst,
}))
# for i in range(N_IMAGES):
#     target_label = target_labels_idx[i]
#     active = u_bool[target_label]
#     print target_label, preds[active, i], preds[:, i]
adv_fail = np.any(np.logical_and(mask, preds != target_labels_idx[None, :]), 0)
target_predict = 1. - np.average(adv_fail.astype(np.float32))
print 'target unanimously predicted', target_predict
unanimous_mask = np.logical_and(mask, np.logical_not(adv_fail))
unanimous_confidence = np.average(conf[unanimous_mask])
print 'unanimous ones\' average confidence', unanimous_confidence
print 'average l2 dist (sqrt)', np.average(np.sqrt(l2d[np.logical_not(adv_fail)]))
print 'failed on', np.where(adv_fail)
print ''

# Select commonly confused examples
common = u_bool[target_labels_idx, orig_labels_idx]
l2d_common = l2d[common]
adv_fail_common = adv_fail[common]
target_predict_common = 1. - np.average(adv_fail_common.astype(np.float32))
print 'common', l2d_common.shape[0]
print 'target unanimously predicted', target_predict_common
print 'average l2 dist (sqrt)', np.average(np.sqrt(l2d_common))
print ''

# Select uncommon
uncommon = np.logical_not(common)
l2d_uncommon = l2d[uncommon]
adv_fail_uncommon = adv_fail[uncommon]
target_predict_uncommon = 1. - np.average(adv_fail_uncommon.astype(np.float32))
print 'uncommon', l2d_uncommon.shape[0]
print 'target unanimously predicted', target_predict_uncommon
print 'average l2 dist (sqrt)', np.average(np.sqrt(l2d_uncommon))
print ''
