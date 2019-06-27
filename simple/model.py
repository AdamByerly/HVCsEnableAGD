# Copyright 2019 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from cnn_helpers import make_batch_norm, make_relu, make_max_pool_2x2
from cnn_helpers import make_conv_3x3_no_bias, make_conv_3x3_stride_2_no_bias
from cnn_helpers import merge_towers_and_optimize
from cnn_helpers import make_fc, make_dropout, make_flatten


def make_tower(x_in, y_out, is_training, count_classes):
    with tf.name_scope("convs"):
        conv1  = make_conv_3x3_stride_2_no_bias("conv1", x_in, 32)
        bn1    = make_batch_norm("bn1", conv1, is_training)
        relu1  = make_relu("relu1", bn1)

        conv2  = make_conv_3x3_no_bias("conv2", relu1, 32)
        bn2    = make_batch_norm("bn2", conv2, is_training)
        relu2  = make_relu("relu2", bn2)

        conv3  = make_conv_3x3_no_bias("conv3", relu2, 32)
        bn3    = make_batch_norm("bn3", conv3, is_training)
        relu3  = make_relu("relu3", bn3)
        pool1  = make_max_pool_2x2("pool1", relu3)

        conv4  = make_conv_3x3_no_bias("conv4", pool1, 64)
        bn4    = make_batch_norm("bn4", conv4, is_training)
        relu4  = make_relu("relu4", bn4)

        conv5  = make_conv_3x3_no_bias("conv5", relu4, 64)
        bn5    = make_batch_norm("bn5", conv5, is_training)
        relu5  = make_relu("relu5", bn5)

        conv6  = make_conv_3x3_no_bias("conv6", relu5, 64)
        bn6    = make_batch_norm("bn6", conv6, is_training)
        relu6  = make_relu("relu6", bn6)
        pool2  = make_max_pool_2x2("pool2", relu6)

        conv7  = make_conv_3x3_no_bias("conv7", pool2, 128)
        bn7    = make_batch_norm("bn7", conv7, is_training)
        relu7  = make_relu("relu7", bn7)

        conv8  = make_conv_3x3_no_bias("conv8", relu7, 128)
        bn8    = make_batch_norm("bn8", conv8, is_training)
        relu8  = make_relu("relu8", bn8)

        conv9  = make_conv_3x3_no_bias("conv9", relu8, 128)
        bn9    = make_batch_norm("bn9", conv9, is_training)
        relu9  = make_relu("relu9", bn9)
        pool3  = make_max_pool_2x2("pool3", relu9)

        conv10 = make_conv_3x3_no_bias("conv10", pool3, 256)
        bn10   = make_batch_norm("bn10", conv10, is_training)
        relu10 = make_relu("relu10", bn10)

        conv11 = make_conv_3x3_no_bias("conv11", relu10, 256)
        bn11   = make_batch_norm("bn11", conv11, is_training)
        relu11 = make_relu("relu11", bn11)

    with tf.name_scope("fcs"):
        flat = make_flatten("flatten", relu11)

        keep_prob = tf.cond(is_training,
                            lambda: tf.constant(0.5),
                            lambda: tf.constant(1.0),
                            name="keep_prob")

        do1 = make_dropout("do1", flat, keep_prob)
        logits = make_fc("fc1", do1, count_classes)

    with tf.name_scope("loss"):
        y_out = tf.stop_gradient(y_out)
        preds = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=y_out)
        loss = tf.reduce_mean(preds)
    return logits, loss


def run_towers(optimizer, global_step, is_training,
        training_data, validation_data, count_classes, num_gpus):
    with tf.device("/device:CPU:0"), tf.name_scope("input/train_or_eval"):
        images, labels = \
            tf.cond(is_training, lambda: training_data, lambda: validation_data)
    labels_list = []
    logits_list = []
    loss_list   = []
    grads       = []
    for i in range(num_gpus):
        tower_name = "tower%d" % i
        with tf.device("/device:GPU:%d" % i):
            with tf.name_scope(tower_name):
                these_logits, this_loss = make_tower(
                    images, labels, is_training, count_classes)
            logits_list.append(these_logits)
            loss_list.append(this_loss)
            labels_list.append(labels)
            grads.append(optimizer.compute_gradients(this_loss))

    train_op, loss, acc_top_1, acc_top_5 = merge_towers_and_optimize(
        optimizer, global_step, grads, logits_list, loss_list, labels_list)

    return train_op, loss, acc_top_1, acc_top_5
