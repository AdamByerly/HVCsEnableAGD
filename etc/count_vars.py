import numpy as np
import tensorflow as tf
# from inception_v3.model_hvc_1 import make_tower # 24,452,528
# from inception_v3.model import make_tower # 24,454,512
# from simple.model import make_tower # 5,464,200
from simple.model_hvc import make_tower # 5,463,216

x = make_tower("tower1",
    tf.zeros([1, 224, 224, 3]),    # input image size (and color channels)
    tf.zeros([1, 1000]),           # classes
    tf.constant(True),
    1000)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    tf.global_variables_initializer().run()

    _, variables = sess.run([x, tf.trainable_variables()])

    var_count = np.array([(var == var).sum() for var in variables]).sum()

    print("Total Variables: {:,}".format(var_count))
