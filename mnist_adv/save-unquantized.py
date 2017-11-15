from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

data_dir = 'mnist_data'
output_dir = 'results/'
n_adv_examples = 100
INPUT_DIM = 28
INPUT_CHANNELS = 1

data = input_data.read_data_sets(data_dir, one_hot=True)
xu = np.reshape(data.test.images, [-1, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS])
xu = xu[:n_adv_examples]
np.save(output_dir + 'unquantized_images.npy', xu)
