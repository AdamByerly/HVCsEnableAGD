# Copyright 2018 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# The above copyright notice is applied in accordance with the license
#  of the codebase from which the following was derived.
# That code is Copyright 2016 Google, Inc. and was retrieved from:
#  https://github.com/tensorflow/models/blob/master
#  /research/inception/inception/slim/losses.py

import tensorflow as tf


def lsr_loss(logits, one_hot_labels,
             label_smoothing=0, weight=1.0, scope=None):
    """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

    It can scale the loss by weight factor, and smooth the labels.

    Args:
      logits: [batch_size, num_classes] logits outputs of the network .
      one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
      label_smoothing: if greater than 0 then smooth the labels.
      weight: scale the loss by this factor.
      scope: Optional scope for name_scope.

    Returns:
      A tensor with the softmax_cross_entropy loss.
    """
    logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
    with tf.name_scope(scope, "lsr_loss", [logits, one_hot_labels]):
        num_classes = one_hot_labels.get_shape()[-1].value
        one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
        if label_smoothing > 0:
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            one_hot_labels = one_hot_labels*smooth_positives+smooth_negatives

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            lables=one_hot_labels, logits=logits, name="xentropy")

        weight = tf.convert_to_tensor(
            weight, dtype=logits.dtype.base_dtype, name="loss_weight")
        return tf.multiply(weight, tf.reduce_mean(cross_entropy), name="value")
