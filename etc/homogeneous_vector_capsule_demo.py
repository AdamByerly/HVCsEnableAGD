import tensorflow as tf

# two "batches" of two tiny capsules, each with 4 dimensions
a = [[[1., 2., 3., 4.],
      [5., 6., 7., 8.]],
     [[1., 2., 3., 4.],
      [5., 6., 7., 8.]]]

# the weights matrix
# (inner two dimensions match the dimensions of the input capsules tensor)
# (3 in the outer dimension which creates 3 output capsules)
# (note that there is one matrix used by all items in the batch)
b = [[[.1, .1, .1, .1],
      [.4, .4, .4, .4]],
     [[.2, .2, .2, .2],
      [.8, .8, .8, .8]],
     [[.3, .3, .3, .3],
      [.5, .5, .5, .5]]]

# primary capsule shape:
#   0: batches
#   1: expansion to allow for the element-wise multiplication to function
#   2: number of primary capsules
#   3: dimensions of each capsule
matrix1 = tf.constant(a, shape=[2, 1, 2, 4])

# weight matrix shape:
#   0: expansion for batches
#   1: number of output capsules to create
#   2: number of input capsules
#   3: dimensions of each capsule
matrix2 = tf.constant(b, shape=[1, 3, 2, 4])

# tile the weight matrix for each item in the batch
matrix2 = tf.tile(matrix2, [2, 1, 1, 1])

# multiply (element-wise) primary capsules by the weight matrix
product = tf.multiply(matrix1, matrix2)

# sum the result and shape into 4D capsules
summed  = tf.reduce_sum(product, 2)

# this is where the batch norm and relu would go

# compute the unscaled predictions
normed  = tf.norm(summed, axis=2)

# and translate into probabilities
probs   = tf.nn.softmax(normed)

with tf.Session() as sess:
    p, s, n, pb = sess.run([product, summed, normed, probs])
    print("output capsule inputs: (shape: {})\n{}\n".format(p.shape, p))
    print("output capsules: (shape: {})\n{}\n".format(s.shape, s))
    print("norm of output capsules: (shape: {})\n{}\n".format(n.shape, n))
    print("softmax probabilities: (shape: {})\n{}\n".format(pb.shape, pb))
