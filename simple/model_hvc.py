import tensorflow as tf
from cnn_helpers import make_batch_norm, make_relu
from cnn_helpers import average_gradients, make_max_pool_2x2
from cnn_helpers import make_conv_3x3_no_bias, make_conv_3x3_stride_2_no_bias


def make_tower(tower_name, x_in, y_out, is_training, count_classes):
    cs     = tower_name+"/convs"  # convs scope

    conv1  = make_conv_3x3_stride_2_no_bias("conv1", cs, x_in, 32)
    bn1    = make_batch_norm("bn1", cs, conv1, is_training)
    relu1  = make_relu("relu1", cs, bn1)

    conv2  = make_conv_3x3_no_bias("conv2", cs, relu1, 32)
    bn2    = make_batch_norm("bn2", cs, conv2, is_training)
    relu2  = make_relu("relu2", cs, bn2)

    conv3  = make_conv_3x3_no_bias("conv3", cs, relu2, 32)
    bn3    = make_batch_norm("bn3", cs, conv3, is_training)
    relu3  = make_relu("relu3", cs, bn3)
    pool1  = make_max_pool_2x2("pool1", cs, relu3)

    conv4  = make_conv_3x3_no_bias("conv4", cs, pool1, 64)
    bn4    = make_batch_norm("bn4", cs, conv4, is_training)
    relu4  = make_relu("relu4", cs, bn4)

    conv5  = make_conv_3x3_no_bias("conv5", cs, relu4, 64)
    bn5    = make_batch_norm("bn5", cs, conv5, is_training)
    relu5  = make_relu("relu5", cs, bn5)

    conv6  = make_conv_3x3_no_bias("conv6", cs, relu5, 64)
    bn6    = make_batch_norm("bn6", cs, conv6, is_training)
    relu6  = make_relu("relu6", cs, bn6)
    pool2  = make_max_pool_2x2("pool2", cs, relu6)

    conv7  = make_conv_3x3_no_bias("conv7", cs, pool2, 128)
    bn7    = make_batch_norm("bn7", cs, conv7, is_training)
    relu7  = make_relu("relu7", cs, bn7)

    conv8  = make_conv_3x3_no_bias("conv8", cs, relu7, 128)
    bn8    = make_batch_norm("bn8", cs, conv8, is_training)
    relu8  = make_relu("relu8", cs, bn8)

    conv9  = make_conv_3x3_no_bias("conv9", cs, relu8, 128)
    bn9    = make_batch_norm("bn9", cs, conv9, is_training)
    relu9  = make_relu("relu9", cs, bn9)
    pool3  = make_max_pool_2x2("pool3", cs, relu9)

    conv10 = make_conv_3x3_no_bias("conv10", cs, pool3, 256)
    bn10   = make_batch_norm("bn10", cs, conv10, is_training)
    relu10 = make_relu("relu10", cs, bn10)

    conv11 = make_conv_3x3_no_bias("conv11", cs, relu10, 256)
    bn11   = make_batch_norm("bn11", cs, conv11, is_training)
    relu11 = make_relu("relu11", cs, bn11)

    with tf.device("/device:CPU:0"),\
            tf.variable_scope("vars/convs", reuse=tf.AUTO_REUSE):
        w_out_cap = tf.get_variable("W_pcap", shape=[1, count_classes, 512, 8],
                        initializer=tf.glorot_uniform_initializer())

    # TODO: put this in a namespace
    pcap      = tf.reshape(relu11, [-1, 1, 512, 8])
    batch_sz  = pcap.get_shape().as_list()[0]
    tiled     = tf.tile(w_out_cap, [batch_sz, 1, 1, 1])
    out_cap   = tf.reduce_sum(tf.multiply(pcap, tiled), 2)
    bn12      = make_batch_norm("bn12", cs, out_cap, is_training)
    relu12    = make_relu("relu12", cs, bn12)

    logits    = tf.norm(relu12, axis=2)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    with tf.name_scope(tower_name + "/loss"):
        y_out = tf.stop_gradient(y_out)
        preds = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=y_out)
        loss  = tf.reduce_mean(preds)
    return logits, preds, loss


def run_towers(is_training, training_data, validation_data, count_classes):
    with tf.device("/device:CPU:0"), tf.name_scope("input/train_or_eval"):
        images, labels = \
            tf.cond(is_training, lambda: training_data, lambda: validation_data)
        images_1, images_2 = tf.split(images, num_or_size_splits=2)
        labels_1, labels_2 = tf.split(labels, num_or_size_splits=2)
    with tf.device("/device:GPU:0"):
        logits1, preds1, loss1 = make_tower("tower1",
            images_1, labels_1, is_training, count_classes)
    with tf.device("/device:GPU:1"):
        logits2, preds2, loss2 = make_tower("tower2",
            images_2, labels_2, is_training, count_classes)
    with tf.device("/device:GPU:1"),\
         tf.name_scope("metrics/concat_tower_outputs"):
        logits = tf.concat([logits1, logits2], 0)
        # preds  = tf.concat([preds1, preds2], 0)
        # tf.summary.histogram('predictions/activations', preds)
        # tf.summary.scalar('predictions/sparsity', tf.nn.zero_fraction(preds))
        # tf.summary.histogram('logits/activations', logits)
        # tf.summary.scalar('logits/sparsity', tf.nn.zero_fraction(logits))
    return loss1, loss2, logits, labels


def apply_gradients(loss1, loss2, global_step, trainer):
    with tf.device("/device:GPU:0"), tf.name_scope("tower1/compute_grads"):
        grads1 = trainer.compute_gradients(loss1)
    with tf.device("/device:GPU:1"), tf.name_scope("tower2/compute_grads"):
        grads2 = trainer.compute_gradients(loss2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.device("/device:GPU:0"), tf.name_scope("merge_grads"):
            grads   = average_gradients([grads1, grads2])
        with tf.device("/device:CPU:0"), tf.name_scope("apply_grads"):
            applied = trainer.apply_gradients(grads, global_step)
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.summary.histogram(var.op.name + '/gradients', grad)
    return applied


def compute_total_loss(loss1, loss2):
    with tf.device("/device:GPU:1"), tf.name_scope("loss"):
        return tf.reduce_mean([loss1, loss2], 0)


def evaluate_validation(logits, labels):
    with tf.device("/device:GPU:1"), tf.name_scope("metrics"):
        labels = tf.argmax(labels, 1)
        in_top_1 = tf.nn.in_top_k(logits, labels, 1)
        in_top_5 = tf.nn.in_top_k(logits, labels, 5)
        acc_top_1 = tf.reduce_mean(tf.cast(in_top_1, tf.float32))
        acc_top_5 = tf.reduce_mean(tf.cast(in_top_5, tf.float32))
        return acc_top_1, acc_top_5
