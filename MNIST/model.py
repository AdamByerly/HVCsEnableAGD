import tensorflow as tf
from cnn_helpers import average_gradients, make_conv_3x3_no_bias
from cnn_helpers import make_batch_norm, make_relu, make_fc_no_bias
from cnn_helpers import make_caps_from_conv, make_homogeneous_vector_caps
from cnn_helpers import make_conv_3x3_stride_2_no_bias, make_fc, make_flatten

BATCH_SIZE                     = 128
RECONSRUCTION_LOSS_WEIGHT      = 0.0005
PCAP_CHANNELS                  = 32
PCAP_OUTPUT_NUM_CAPSULES       = 1152
PCAP_OUTPUT_DIM_CAPSULES       = 8
DIGITCAPS_NUM_CAPSULES         = 10  # no. of classes
DIGITCAPS_DIM_CAPSULES         = 16


def mask_label(tower_name, tensor, mask):
    with tf.name_scope(tower_name + "/mask"):
        x = tensor * tf.expand_dims(mask, -1)
        return tf.reshape(x,
            tf.stack([-1, tf.reduce_prod(tf.shape(x)[1:])]))


def make_tower(tower_name, x_in, y_out, is_training):
    cs      = tower_name+"/convs"  # convs scope
    cs2     = tower_name+"/caps"   # caps scope
    fcs     = tower_name+"/fcs"    # fully connecteds scope

    conv1   = make_conv_3x3_stride_2_no_bias("conv1", cs, x_in, 128)
    bn1     = make_batch_norm("bn1", cs, conv1, is_training)
    relu1   = make_relu("relu1", cs, bn1)

    conv2   = make_conv_3x3_no_bias("conv2", cs, relu1, 128)
    bn2     = make_batch_norm("bn2", cs, conv2, is_training)
    relu2   = make_relu("relu2", cs, bn2)

    conv3   = make_conv_3x3_no_bias("conv3", cs, relu2, 128)
    bn3     = make_batch_norm("bn3", cs, conv3, is_training)
    relu3   = make_relu("relu3", cs, bn3)

    conv4   = make_conv_3x3_no_bias("conv4", cs, relu3, 256)
    bn4     = make_batch_norm("bn4", cs, conv4, is_training)
    relu4   = make_relu("relu4", cs, bn4)

    conv5   = make_conv_3x3_no_bias("conv5", cs, relu4, 256)
    bn5     = make_batch_norm("bn5", cs, conv5, is_training)
    relu5   = make_relu("relu5", cs, bn5)

    conv6   = make_conv_3x3_no_bias("conv6", cs, relu5, 256)
    bn6     = make_batch_norm("bn6", cs, conv6, is_training)
    relu6   = make_relu("relu6", cs, bn6)

    flat     = make_flatten("flatten", cs, relu6)
    fc1      = make_fc_no_bias("fc1", fcs, flat, 512)
    bn_fc1   = make_batch_norm("bn_fc1", fcs, fc1, is_training)
    relu_fc1 = make_relu("relu_fc1", fcs, bn_fc1)
    logits   = make_fc("fc2", fcs, relu_fc1, 10)

    # pcap    = make_caps_from_conv(
    #             "pcap", cs2, relu6, 8, 288, int(BATCH_SIZE/2))
    # out_cap = make_homogeneous_vector_caps(
    #             "out_cap", cs2, pcap, 10, 8, int(BATCH_SIZE/2))
    # bn12    = make_batch_norm("bn12", cs, out_cap, is_training)
    # relu12  = make_relu("relu12", cs, bn12)
    #
    # logits  = tf.norm(relu12, axis=2)

    # masked_by_y = mask_label(tower_name, out_cap, y_out)
    #
    # fc1      = make_fc_no_bias("fc1", tower_name + "/decoder/fc_1",
    #             masked_by_y, 512)
    # bn_fc1   = make_batch_norm("bn_fc1", tower_name + "/decoder/fc_1",
    #             fc1, is_training)
    # relu_fc1 = make_relu("relu_fc1", tower_name + "/decoder/fc_1", bn_fc1)
    #
    # fc2      = make_fc_no_bias("fc2", tower_name + "/decoder/fc_2",
    #             relu_fc1, 1024)
    # bn_fc2   = make_batch_norm("bn_fc2", tower_name + "/decoder/fc_2",
    #             fc2, is_training)
    # relu_fc2 = make_relu("relu_fc2", tower_name + "/decoder/fc_2", bn_fc2)
    #
    # fc3      = make_fc_no_bias("fc3", tower_name + "/decoder/fc_3",
    #             relu_fc2, 784)
    # bn_fc3   = make_batch_norm("bn_fc3", tower_name + "/decoder/fc_3",
    #             fc3, is_training)
    # sgmd_fc3 = make_sigmoid("relu_fc3", tower_name + "/decoder/fc_3", bn_fc3)
    #
    # with tf.name_scope(tower_name + "/out_recon"):
    #     reconed  = tf.reshape(sgmd_fc3, shape=[-1, 28, 28, 1])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    with tf.name_scope(tower_name + "/loss"):
        y_out = tf.stop_gradient(y_out)
        preds = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=y_out)
        loss_labels = tf.reduce_mean(preds)
        loss_recon = tf.constant(0.0)
        # loss_recon = RECONSRUCTION_LOSS_WEIGHT *\
        #                 tf.losses.mean_squared_error(x_in,
        #                 reconed, reduction=
        #                 tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss_total = loss_labels + loss_recon

    return logits, preds, loss_labels, loss_recon, loss_total


def run_towers(iterator, is_training):
    with tf.device("/device:CPU:0"), tf.name_scope("input/"):
        images, labels = iterator.get_next()
        images_1, images_2 = tf.split(images, num_or_size_splits=2)
        labels_1, labels_2 = tf.split(labels, num_or_size_splits=2)
    with tf.device("/device:GPU:0"):
        logits1, preds1, loss_labels1, loss_recon1, loss_total1 =\
            make_tower("tower1", images_1, labels_1, is_training)
    with tf.device("/device:GPU:1"):
        logits2, preds2, loss_labels2, loss_recon2, loss_total2 =\
            make_tower("tower2", images_2, labels_2, is_training)
    with tf.device("/device:GPU:1"),\
            tf.name_scope("metrics/concat_tower_outputs"):
        logits = tf.concat([logits1, logits2], 0)
        # preds  = tf.concat([preds1, preds2], 0)
        # tf.summary.histogram('predictions/activations', preds)
        # tf.summary.scalar('predictions/sparsity', tf.nn.zero_fraction(preds))
        # tf.summary.histogram('logits/activations', logits)
        # tf.summary.scalar('logits/sparsity', tf.nn.zero_fraction(logits))
    return loss_labels1, loss_labels2, loss_recon1, loss_recon2, \
           loss_total1, loss_total2, logits, labels


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


def compute_total_loss(loss_labels1, loss_labels2,
        loss_recon1, loss_recon2, loss_total1, loss_total2):
    with tf.device("/device:GPU:1"), tf.name_scope("loss"):
        ll = tf.reduce_mean([loss_labels1, loss_labels2], 0)
        lr = tf.reduce_mean([loss_recon1, loss_recon2], 0)
        lt = tf.reduce_mean([loss_total1, loss_total2], 0)
        return ll, lr, lt


def evaluate_validation(logits, labels):
    with tf.device("/device:GPU:1"), tf.name_scope("metrics"):
        prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        return tf.reduce_mean(tf.cast(prediction, tf.float32))
