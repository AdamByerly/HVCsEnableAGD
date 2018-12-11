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


def make_conv_no_bias(op_name, op_scope, in_tensor, filter_size_h,
        filter_size_w, filters, strides=(1, 1, 1, 1), padding="VALID"):
    input_size = in_tensor.get_shape().as_list()[3]
    shape = (filter_size_h, filter_size_w, input_size, filters)
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_"+op_name, shape=shape, regularizer=l2reg,
                initializer=tf.glorot_uniform_initializer())
    with tf.name_scope(op_scope):
        return tf.nn.conv2d(in_tensor, w,
                strides=strides, padding=padding, name=op_name)


def make_conv(op_name, op_scope, in_tensor, filter_size_h,
        filter_size_w, filters, strides=(1, 1, 1, 1), padding="VALID"):
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        b = tf.get_variable("b_"+op_name, shape=filters,
                initializer=tf.zeros_initializer())
    with tf.name_scope(op_scope):
        conv = make_conv_no_bias(op_name, op_scope, in_tensor,
                filter_size_h, filter_size_w, filters, strides, padding)
        return tf.add(conv, b)


def make_conv_3x3(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv(op_name, op_scope,
        in_tensor, 3, 3, filters, strides, padding)


def make_conv_9x9(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv(op_name, op_scope,
        in_tensor, 9, 9, filters, strides, padding)


def make_conv_9x9_stride_2(op_name, op_scope, in_tensor,
        filters, padding="VALID"):
    return make_conv(op_name, op_scope,
        in_tensor, 9, 9, filters, (1, 2, 2, 1), padding)


def make_conv_1x1_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 1, 1, filters, strides, padding)


def make_conv_3x1_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 3, 1, filters, strides, padding)


def make_conv_1x3_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 1, 3, filters, strides, padding)


def make_conv_3x3_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 3, 3, filters, strides, padding)


def make_conv_5x5_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 5, 5, filters, strides, padding)


def make_conv_7x1_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 7, 1, filters, strides, padding)


def make_conv_1x7_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 1, 7, filters, strides, padding)


def make_conv_9x9_no_bias(op_name, op_scope, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 9, 9, filters, strides, padding)


def make_conv_3x3_stride_2_no_bias(op_name, op_scope, in_tensor,
        filters, padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 3, 3, filters, (1, 2, 2, 1), padding)


def make_conv_9x9_stride_2_no_bias(op_name, op_scope, in_tensor,
        filters, padding="VALID"):
    return make_conv_no_bias(op_name, op_scope,
        in_tensor, 9, 9, filters, (1, 2, 2, 1), padding)


def make_relu(op_name, op_scope, in_tensor):
    with tf.name_scope(op_scope):
        return tf.nn.relu(in_tensor, name=op_name)


def make_sigmoid(op_name, op_scope, in_tensor):
    with tf.name_scope(op_scope):
        return tf.nn.relu(in_tensor, name=op_name)


def make_avg_pool(op_name, op_scope,
        in_tensor, window_size_h, window_size_w, padding="VALID"):
    with tf.name_scope(op_scope):
        return tf.nn.pool(input=in_tensor,
            window_shape=[window_size_h, window_size_w],
            strides=[2, 2], pooling_type="AVG", padding=padding, name=op_name)


def make_max_pool_2x2(op_name, op_scope, in_tensor, padding="VALID"):
    with tf.name_scope(op_scope):
        return tf.nn.pool(input=in_tensor, window_shape=[2, 2],
            strides=[2, 2], pooling_type="MAX", padding=padding, name=op_name)


def make_max_pool_3x3(op_name, op_scope, in_tensor, padding="VALID"):
    with tf.name_scope(op_scope):
        return tf.nn.pool(input=in_tensor, window_shape=[3, 3],
            strides=[2, 2], pooling_type="MAX", padding=padding, name=op_name)


def make_avg_pool_3x3(op_name, op_scope, in_tensor, padding="VALID"):
    with tf.name_scope(op_scope):
        return tf.nn.pool(input=in_tensor, window_shape=[3, 3],
            strides=[2, 2], pooling_type="AVG", padding=padding, name=op_name)


def make_avg_pool_5x5_stride_3(op_name, op_scope, in_tensor, padding="VALID"):
    with tf.name_scope(op_scope):
        return tf.nn.pool(input=in_tensor, window_shape=[5, 5],
            strides=[3, 3], pooling_type="AVG", padding=padding, name=op_name)


def make_flatten(op_name, op_scope, in_tensor):
    with tf.name_scope(op_scope):
        input_size = in_tensor.get_shape().as_list()
        size = input_size[1]*input_size[2]*input_size[3]
        return tf.reshape(in_tensor, [-1, size], name=op_name)


def make_fc(op_name, op_scope, in_tensor, neurons):
    input_size = in_tensor.get_shape().as_list()[1]
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/fcs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_"+op_name, shape=[input_size, neurons],
                regularizer=l2reg, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b_"+op_name, shape=[neurons],
                initializer=tf.zeros_initializer())
    with tf.name_scope(op_scope):
        fc1 = tf.matmul(in_tensor, w, name="matmul_"+op_name)
        return tf.add(fc1, b, name="add_"+op_name)


def make_fc_no_bias(op_name, op_scope, in_tensor, neurons):
    input_size = in_tensor.get_shape().as_list()[1]
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/fcs", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("W_"+op_name, shape=[input_size, neurons],
                regularizer=l2reg, initializer=tf.glorot_uniform_initializer())
    with tf.name_scope(op_scope):
        return tf.matmul(in_tensor, w, name="matmul_"+op_name)


def make_caps_from_conv(op_name, op_scope,
        in_tensor, cap_dims, cap_count, batch_size):
    with tf.name_scope(op_scope):
        return tf.reshape(in_tensor,
            [batch_size, 1, cap_count, cap_dims], name="caps_shape_"+op_name)


def make_homogeneous_vector_caps(op_name,
        op_scope, in_tensor, out_caps, cap_dims, batch_size):
    with tf.device("/device:CPU:0"),\
            tf.variable_scope("vars/caps", reuse=tf.AUTO_REUSE):
        in_caps_sz = in_tensor.get_shape().as_list()[2]
        w_out_cap  = tf.get_variable("w_"+op_name,
                        shape=[1, out_caps, in_caps_sz, cap_dims],
                        initializer=tf.glorot_uniform_initializer())
    with tf.name_scope(op_scope):
        tiled      = tf.tile(w_out_cap, [batch_size, 1, 1, 1])
        return tf.reduce_sum(tf.multiply(in_tensor, tiled), 2)


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
