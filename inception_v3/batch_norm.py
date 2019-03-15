# Copyright 2016 Google Inc. All Rights Reserved.
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
#
# The above copyright notice is applied in accordance with the license
#  of the codebase from which the following was derived and retrieved from:
#    https://github.com/tensorflow/models/blob/master
#      /research/inception/inception/slim/ops.py

import tensorflow as tf
from tensorflow.python.training import moving_averages


def batch_norm(scope, inputs, is_training=True, decay=0.9997, epsilon=0.001):
    """Adds a Batch Normalization layer.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels]
                or [batch_size, channels].
        decay: decay for the moving average.
        epsilon: small float added to variance to avoid dividing by zero.
        is_training: whether or not the model is in training mode.
        scope: Optional scope for variable_scope.
    Returns:
        a tensor representing the output of the operation.
    """
    with tf.device("/device:CPU:0"),\
            tf.variable_scope(scope, "batch_norm",
                [inputs], reuse=tf.AUTO_REUSE):
        inputs_shape       = inputs.get_shape()
        axis               = list(range(len(inputs_shape) - 1))
        params_shape       = inputs_shape[-1:]
        beta               = tf.get_variable("beta", shape=params_shape,
                                initializer=tf.zeros_initializer())
        moving_collections = [tf.GraphKeys.GLOBAL_VARIABLES,
                              tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
        moving_mean        = tf.get_variable("moving_mean", params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False, collections=moving_collections)
        moving_variance    = tf.get_variable("moving_variance", params_shape,
                                initializer=tf.ones_initializer(),
                                trainable=False, collections=moving_collections)
    with tf.name_scope(scope):
        def training_func():
            # Calculate the moments based on the individual batch.
            mean, variance         = tf.nn.moments(inputs, axis)
            update_moving_mean     = moving_averages.assign_moving_average(
                                        moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(
                                        moving_variance, variance, decay)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                 update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                 update_moving_variance)
            return mean, variance

        def inferring_func():
            # Just use the moving_mean and moving_variance.
            return moving_mean, moving_variance

        mean, variance = tf.cond(is_training,
                            lambda: training_func(),
                            lambda: inferring_func())

        # Normalize the activations.
        outputs = tf.nn.batch_normalization(inputs,
                    mean, variance, beta, None, epsilon)
        outputs.set_shape(inputs.get_shape())
        return outputs
