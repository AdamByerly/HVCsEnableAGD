import numpy as np
import tensorflow as tf
from model2 import make_tower

tf.reset_default_graph()

with tf.name_scope("input/placeholders"):
    keep_prob   = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

logits, preds, loss = make_tower("tower1",
    tf.zeros([64, 224, 224, 3]),
    tf.zeros([64, 1000]),
    keep_prob, is_training, 1000)

with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True)) as sess:

    tf.global_variables_initializer().run()

    _, _, _, variables = sess.run(
        [logits, preds, loss, tf.trainable_variables()],
        feed_dict={keep_prob: 0.5, is_training: True})

    var_count = np.array([(var == var).sum() for var in variables]).sum()

    print(var_count)
