import numpy as np
import tensorflow as tf
from MNIST.model import make_tower

x = make_tower("tower1",
    tf.zeros([64, 28, 28, 1]),    # input image size (and color channels)
    tf.zeros([64, 10]),           # classes
    True)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    tf.global_variables_initializer().run()

    _, variables = sess.run([x, tf.trainable_variables()])

    var_count = np.array([(var == var).sum() for var in variables]).sum()

    print("Total Variables: {:,}".format(var_count))
