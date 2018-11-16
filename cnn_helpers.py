import tensorflow as tf


L2_REG_BETA = 0.0005


def l2reg(t):
    return L2_REG_BETA * tf.nn.l2_loss(t)


# TODO: figure out how to not have to reuse these variables
def make_batch_norm(op_name, op_scope, in_tensor, is_train):
    with tf.name_scope(op_scope),\
            tf.variable_scope("vars/bns/"+op_name, reuse=tf.AUTO_REUSE):
        return tf.layers.batch_normalization(in_tensor, momentum=0.99,
            training=is_train)


def make_conv_3x3(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    input_size = in_tensor.get_shape().as_list()[3]
    shape = (3, 3, input_size, filters)
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_" + op_name, shape=shape, regularizer=l2reg,
                initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b_" + op_name, shape=shape[-1],
                initializer=tf.zeros_initializer())
    with tf.name_scope(op_scope):
        conv = tf.nn.conv2d(in_tensor, w,
                strides=strides, padding=padding, name=op_name)
        return tf.add(conv, b)


def make_conv_3x3_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    input_size = in_tensor.get_shape().as_list()[3]
    shape = (3, 3, input_size, filters)
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_" + op_name, shape=shape, regularizer=l2reg,
                initializer=tf.glorot_uniform_initializer())
    with tf.name_scope(op_scope):
        return tf.nn.conv2d(in_tensor, w,
                strides=strides, padding=padding, name=op_name)


def make_conv_3x3_stride_2_no_bias(op_name, op_scope, in_tensor,
        filters, padding="VALID"):
    return make_conv_3x3_no_bias(op_name, op_scope,
        in_tensor, filters, (1, 2, 2, 1), padding)


def make_relu(op_name, op_scope, in_tensor):
    with tf.name_scope(op_scope):
        return tf.nn.relu(in_tensor, name=op_name)


def make_max_pool_2x2(op_name, op_scope, in_tensor, padding="VALID"):
    with tf.name_scope(op_scope):
        return tf.nn.pool(input=in_tensor, window_shape=[2, 2],
            strides=[2, 2], pooling_type="MAX", padding=padding, name=op_name)


def make_flatten(op_name, op_scope, in_tensor):
    with tf.name_scope(op_scope):
        input_size = in_tensor.get_shape().as_list()
        size = input_size[1]*input_size[2]*input_size[3]
        return tf.reshape(in_tensor, [-1, size], name=op_name)


def make_fc(op_name, op_scope, in_tensor, neurons):
    input_size = in_tensor.get_shape().as_list()[1]
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/fcs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_" + op_name, shape=[input_size, neurons],
                regularizer=l2reg, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b_" + op_name, shape=[neurons],
                initializer=tf.zeros_initializer())
    with tf.name_scope(op_scope):
        fc1 = tf.matmul(in_tensor, w, name="matmul_" + op_name)
        return tf.add(fc1, b, name="add_" + op_name)


def make_fc_no_bias(op_name, op_scope, in_tensor, neurons):
    input_size = in_tensor.get_shape().as_list()[1]
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/fcs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_" + op_name, shape=[input_size, neurons],
                regularizer=l2reg, initializer=tf.glorot_uniform_initializer())
    with tf.name_scope(op_scope):
        return tf.matmul(in_tensor, w, name="matmul_" + op_name)


def make_dropout(op_name, op_scope, in_tensor, keep_prob):
    with tf.name_scope(op_scope):
        return tf.nn.dropout(in_tensor, keep_prob, name=op_name)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
