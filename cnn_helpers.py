import tensorflow as tf


def l2_regularizer(weight):
    def regularizer(t):
        return weight * tf.nn.l2_loss(t)
    return regularizer


def make_batch_norm(op_name, in_tensor, is_train):
    with tf.name_scope(op_name),\
            tf.variable_scope("vars/bns/"+op_name, reuse=tf.AUTO_REUSE):
        return tf.layers.batch_normalization(in_tensor, momentum=0.99,
            training=is_train)


def make_conv(op_name, in_tensor, filter_size_h,
        filter_size_w, filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        input_size = in_tensor.get_shape().as_list()[3]
        shape = (filter_size_h, filter_size_w, input_size, filters)
        w = tf.get_variable("W_"+op_name, shape=shape,
                regularizer=l2_regularizer(weight_decay),
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable("b_"+op_name, shape=filters,
                initializer=tf.zeros_initializer())
    with tf.name_scope(op_name):
        return tf.add(tf.nn.conv2d(in_tensor, w,
            strides=strides, padding=padding, name=op_name), b)


def make_conv_no_bias(op_name, in_tensor, filter_size_h,
        filter_size_w, filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        input_size = in_tensor.get_shape().as_list()[3]
        shape = (filter_size_h, filter_size_w, input_size, filters)
        w = tf.get_variable("W_"+op_name, shape=shape,
                regularizer=l2_regularizer(weight_decay),
                initializer=tf.truncated_normal_initializer(stddev=stddev))
    with tf.name_scope(op_name):
        return tf.nn.conv2d(in_tensor, w,
            strides=strides, padding=padding, name=op_name)


def make_conv_3x3(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv(op_name,
        in_tensor, 3, 3, filters, strides, padding, weight_decay, stddev)


def make_conv_9x9(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv(op_name,
        in_tensor, 9, 9, filters, strides, padding, weight_decay, stddev)


def make_conv_9x9_stride_2(op_name, in_tensor,
        filters, padding="VALID", weight_decay=0.0005, stddev=0.1):
    return make_conv(op_name,
        in_tensor, 9, 9, filters, (1, 2, 2, 1), padding, weight_decay, stddev)


def make_conv_1x1_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 1, 1, filters, strides, padding, weight_decay, stddev)


def make_conv_3x1_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 3, 1, filters, strides, padding, weight_decay, stddev)


def make_conv_1x3_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 1, 3, filters, strides, padding, weight_decay, stddev)


def make_conv_3x3_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 3, 3, filters, strides, padding, weight_decay, stddev)


def make_conv_5x5_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 5, 5, filters, strides, padding, weight_decay, stddev)


def make_conv_7x1_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 7, 1, filters, strides, padding, weight_decay, stddev)


def make_conv_1x7_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 1, 7, filters, strides, padding, weight_decay, stddev)


def make_conv_9x9_no_bias(op_name, in_tensor,
        filters, strides=(1, 1, 1, 1), padding="VALID",
        weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 9, 9, filters, strides, padding, weight_decay, stddev)


def make_conv_3x3_stride_2_no_bias(op_name, in_tensor,
        filters, padding="VALID", weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 3, 3, filters, (1, 2, 2, 1), padding, weight_decay, stddev)


def make_conv_9x9_stride_2_no_bias(op_name, in_tensor,
        filters, padding="VALID", weight_decay=0.0005, stddev=0.1):
    return make_conv_no_bias(op_name,
        in_tensor, 9, 9, filters, (1, 2, 2, 1), padding, weight_decay, stddev)


def make_max_pool_2x2(op_name, in_tensor, padding="VALID"):
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor, window_shape=[2, 2],
            strides=[2, 2], pooling_type="MAX", padding=padding, name=op_name)


def make_max_pool_3x3(op_name, in_tensor, padding="VALID"):
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor, window_shape=[3, 3],
            strides=[2, 2], pooling_type="MAX", padding=padding, name=op_name)


def make_avg_pool(op_name,
        in_tensor, window_size_h, window_size_w, padding="VALID"):
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor,
            window_shape=[window_size_h, window_size_w],
            strides=[2, 2], pooling_type="AVG", padding=padding, name=op_name)


def make_avg_pool_3x3(op_name, in_tensor, padding="VALID"):
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor, window_shape=[3, 3],
            strides=[3, 3], pooling_type="AVG", padding=padding, name=op_name)


def make_avg_pool_3x3_stride_1(op_name, in_tensor, padding="VALID"):
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor, window_shape=[3, 3],
            strides=[1, 1], pooling_type="AVG", padding=padding, name=op_name)


def make_avg_pool_5x5_stride_3(op_name, in_tensor, padding="VALID"):
    with tf.name_scope(op_name):
        return tf.nn.pool(input=in_tensor, window_shape=[5, 5],
            strides=[3, 3], pooling_type="AVG", padding=padding, name=op_name)


def make_flatten(op_name, in_tensor):
    with tf.name_scope(op_name):
        input_size = tf.shape(in_tensor)
        size = input_size[1]*input_size[2]*input_size[3]
        return tf.reshape(in_tensor, [-1, size], name=op_name)


def make_concat(op_name, axis, in_tensors):
    with tf.name_scope(op_name):
        return tf.concat(axis=axis, values=in_tensors, name=op_name)


def make_relu(op_name, in_tensor):
    with tf.name_scope(op_name):
        return tf.nn.relu(in_tensor, name=op_name)


def make_fc(op_name, in_tensor, neurons,
        weight_decay=0.0005, stddev=0.01):
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/fcs", reuse=tf.AUTO_REUSE):
        input_size = in_tensor.get_shape().as_list()[1]
        w = tf.get_variable("W_"+op_name, shape=[input_size, neurons],
                regularizer=l2_regularizer(weight_decay),
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable("b_"+op_name, shape=[neurons],
                initializer=tf.zeros_initializer())
    with tf.name_scope(op_name):
        fc1 = tf.matmul(in_tensor, w, name="matmul_"+op_name)
        return tf.add(fc1, b, name="add_"+op_name)


def make_fc_no_bias(op_name, in_tensor, neurons,
        weight_decay=0.0005, stddev=0.01):
    with tf.device("/device:CPU:0"),\
         tf.variable_scope("vars/fcs", reuse=tf.AUTO_REUSE):
        input_size = in_tensor.get_shape().as_list()[1]
        w = tf.get_variable("W_"+op_name, shape=[input_size, neurons],
                regularizer=l2_regularizer(weight_decay),
                initializer=tf.truncated_normal_initializer(stddev=stddev))
    with tf.name_scope(op_name):
        return tf.matmul(in_tensor, w, name="matmul_"+op_name)


def make_caps_from_conv(op_name, in_tensor, cap_dims, cap_count):
    with tf.name_scope(op_name):
        return tf.reshape(in_tensor,
            [-1, 1, cap_count, cap_dims], name="caps_shape_"+op_name)


def make_homogeneous_vector_caps(op_name,
        in_tensor, out_caps, cap_dims, weight_decay=0.0005):
    with tf.device("/device:CPU:0"),\
            tf.variable_scope("vars/caps", reuse=tf.AUTO_REUSE):
        in_caps_sz = in_tensor.get_shape().as_list()[2]
        w_out_cap  = tf.get_variable("w_"+op_name,
                        shape=[out_caps, in_caps_sz, cap_dims],
                        regularizer=l2_regularizer(weight_decay),
                        initializer=tf.glorot_uniform_initializer())
    with tf.name_scope(op_name):
        return tf.reduce_sum(tf.multiply(in_tensor, w_out_cap), 2)


def make_norm(op_name, in_tensor):
    with tf.name_scope(op_name):
        return tf.norm(in_tensor, axis=2, name=op_name)


def make_dropout(op_name, in_tensor, keep_prob):
    return tf.nn.dropout(in_tensor, keep_prob, name=op_name)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def merge_towers_and_optimize(optimizer, global_step,
        grads, logits_list, loss_list, labels_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.device("/device:CPU:0"):
            with tf.name_scope("merge_grads"):
                all_grads = average_gradients(grads)
            with tf.name_scope("apply_grads"):
                train_op  = optimizer.apply_gradients(all_grads, global_step)
    with tf.device("/device:GPU:0"), tf.name_scope("metrics"):
        logits = tf.concat(logits_list, 0)
        loss = tf.reduce_mean(loss_list, 0)
        labels = tf.concat(labels_list, 0)
        labels = tf.argmax(labels, 1)
        in_top_1 = tf.nn.in_top_k(logits, labels, 1)
        in_top_5 = tf.nn.in_top_k(logits, labels, 5)
        acc_top_1 = tf.reduce_mean(tf.cast(in_top_1, tf.float32))
        acc_top_5 = tf.reduce_mean(tf.cast(in_top_5, tf.float32))

    return train_op, loss, acc_top_1, acc_top_5
