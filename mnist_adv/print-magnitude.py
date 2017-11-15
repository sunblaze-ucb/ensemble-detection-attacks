import numpy as np

output_dir = 'results/'

x = np.load(output_dir + 'original_images.npy')
x_adv = np.load(output_dir + 'adversarial_images.npy')

l2dist = np.sum(np.square(x_adv - x), (1, 2, 3))
print l2dist
print 'average', np.average(l2dist)

xu = np.load(output_dir + 'unquantized_images.npy')
l2dist_u = np.sum(np.square(x_adv - xu), (1, 2, 3))
print 'average (from unquantized)', np.average(l2dist_u)
