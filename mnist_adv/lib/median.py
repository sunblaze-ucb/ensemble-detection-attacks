import tensorflow as tf

# http://stackoverflow.com/a/43554072/1864688

def pad_amount(k):
    added = k - 1
    # note: this imitates scipy, which puts more at the beginning
    end = added // 2
    start = added - end
    return [start, end]

def neighborhood(x, kh, kw):
    # input: N, H, W, C
    # output: N, H, W, KH, KW, C
    # padding is REFLECT
    xs = tf.shape(x)
    x_pad = tf.pad(x, ([0, 0], pad_amount(kh), pad_amount(kw), [0, 0]), 'SYMMETRIC')
    return tf.reshape(tf.extract_image_patches(x_pad,
                                               [1, kh, kw, 1],
                                               [1, 1, 1, 1],
                                               [1, 1, 1, 1],
                                               'VALID'),
                      (xs[0], xs[1], xs[2], kh, kw, xs[3]))

def median_filter(x, kh, kw):
    neigh_size = kh * kw
    xs = tf.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = tf.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = tf.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    x_top, _ = tf.nn.top_k(x_neigh, rank)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))
