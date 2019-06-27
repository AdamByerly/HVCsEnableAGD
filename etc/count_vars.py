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

import numpy as np
import tensorflow as tf
from simple.model import make_tower as simple_make_tower
from simple.model_hvc import make_tower as simple_hvc_make_tower
from inception_v3.model import make_tower as inception_v3_make_tower
from inception_v3.model_hvc import make_tower as inception_v3_hvc_make_tower


def get_var_count(creator, image_size):
    tf.reset_default_graph()
    model = creator(tf.zeros([1, image_size, image_size, 3]),
            tf.zeros([1, 1000]), tf.constant(True), 1000)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.global_variables_initializer().run()
        _, variables = sess.run([model, tf.trainable_variables()])
        var_count = np.array([(var == var).sum() for var in variables]).sum()
        return var_count


simple_var_count = get_var_count(simple_make_tower, 224)
simple_hvc_var_count = get_var_count(simple_hvc_make_tower, 224)
inceptionv3_var_count = get_var_count(inception_v3_make_tower, 299)
inceptionv3_hvc_var_count = get_var_count(inception_v3_hvc_make_tower, 299)

print("Simple Var. Count ............:  {:,}".format(simple_var_count))
print("Simple w/HVCs Var. Count......:  {:,}".format(simple_hvc_var_count))
print("Inception v3 Var. Count.......: {:,}".format(inceptionv3_var_count))
print("Inception v3 w/HVCs Var. Count: {:,}".format(inceptionv3_hvc_var_count))
