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
from cnn_helpers import make_conv_no_bias, make_conv_1x1_no_bias
from cnn_helpers import make_conv_3x3_no_bias, make_conv_3x3_stride_2_no_bias
from cnn_helpers import make_conv_3x1_no_bias, make_conv_1x3_no_bias
from cnn_helpers import make_conv_5x5_no_bias, make_conv_1x7_no_bias
from cnn_helpers import make_conv_7x1_no_bias, make_avg_pool_3x3_stride_1
from cnn_helpers import make_max_pool_3x3, make_avg_pool_5x5_stride_3
from cnn_helpers import make_avg_pool, make_relu, make_concat
from cnn_helpers import make_flatten, make_fc, make_dropout
from cnn_helpers import merge_towers_and_optimize
from inception_v3.lsr_loss import lsr_loss
from inception_v3.batch_norm import batch_norm

DECAY_RATE = 0.00004


def make_tower(x_in, y_out, is_training, count_classes):
    with tf.name_scope("input_stem"):
        conv1  = make_conv_3x3_stride_2_no_bias("conv1",
                    x_in, 32, weight_decay=DECAY_RATE)
        bn1    = batch_norm("bn1", conv1, is_training)
        relu1  = make_relu("relu1", bn1)

        conv2  = make_conv_3x3_no_bias("conv2",
                    relu1, 32, weight_decay=DECAY_RATE)
        bn2    = batch_norm("bn2", conv2, is_training)
        relu2  = make_relu("relu2", bn2)

        conv3  = make_conv_3x3_no_bias("conv3",
                    relu2, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn3    = batch_norm("bn3", conv3, is_training)
        relu3  = make_relu("relu3", bn3)
        pool3  = make_max_pool_3x3("pool3", relu3)

        conv4  = make_conv_1x1_no_bias("conv4",
                    pool3, 80, weight_decay=DECAY_RATE)
        bn4    = batch_norm("bn4", conv4, is_training)
        relu4  = make_relu("relu4", bn4)

        conv5  = make_conv_3x3_no_bias("conv5",
                    relu4, 192, weight_decay=DECAY_RATE)
        bn5    = batch_norm("bn5", conv5, is_training)
        relu5  = make_relu("relu5", bn5)
        pool5  = make_max_pool_3x3("pool5", relu5)

    with tf.name_scope("mixed_35x35x256a"):
        intnsr = pool5

        # branch1x1
        conv6  = make_conv_1x1_no_bias("conv6",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn6    = batch_norm("bn6", conv6, is_training)
        relu6  = make_relu("relu6", bn6)

        # branch5x5
        conv7  = make_conv_1x1_no_bias("conv7",
                    intnsr, 48, padding="SAME", weight_decay=DECAY_RATE)
        bn7    = batch_norm("bn7", conv7, is_training)
        relu7  = make_relu("relu7", bn7)
        conv8  = make_conv_5x5_no_bias("conv8",
                    relu7, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn8    = batch_norm("bn8", conv8, is_training)
        relu8  = make_relu("relu8", bn8)

        # branch3x3dbl
        conv9  = make_conv_1x1_no_bias("conv9",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn9    = batch_norm("bn9", conv9, is_training)
        relu9  = make_relu("relu9", bn9)
        conv10 = make_conv_3x3_no_bias("conv10",
                    relu9, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn10   = batch_norm("bn10", conv10, is_training)
        relu10 = make_relu("relu10", bn10)
        conv11 = make_conv_3x3_no_bias("conv11",
                    relu10, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn11   = batch_norm("bn11", conv11, is_training)
        relu11 = make_relu("relu11", bn11)

        # branch_pool
        pool12 = make_avg_pool_3x3_stride_1("pool12", intnsr, padding="SAME")
        conv12 = make_conv_1x1_no_bias("conv12",
                    pool12, 32, padding="SAME", weight_decay=DECAY_RATE)
        bn12   = batch_norm("bn12", conv12, is_training)
        relu12 = make_relu("relu12", bn12)

        cc1    = make_concat("cc1", 3, [relu6, relu8, relu11, relu12])

    with tf.name_scope("mixed_35x35x288a"):
        intnsr = cc1

        # branch1x1
        conv13 = make_conv_1x1_no_bias("conv13",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn13   = batch_norm("bn13", conv13, is_training)
        relu13 = make_relu("relu13", bn13)

        # branch5x5
        conv14 = make_conv_1x1_no_bias("conv14",
                    intnsr, 48, padding="SAME", weight_decay=DECAY_RATE)
        bn14   = batch_norm("bn14", conv14, is_training)
        relu14 = make_relu("relu14", bn14)
        conv15 = make_conv_5x5_no_bias("conv15",
                    relu14, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn15   = batch_norm("bn15", conv15, is_training)
        relu15 = make_relu("relu15", bn15)

        # branch3x3dbl
        conv16 = make_conv_1x1_no_bias("conv16",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn16   = batch_norm("bn16", conv16, is_training)
        relu16 = make_relu("relu16", bn16)
        conv17 = make_conv_3x3_no_bias("conv17",
                    relu16, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn17   = batch_norm("bn17", conv17, is_training)
        relu17 = make_relu("relu17", bn17)
        conv18 = make_conv_3x3_no_bias("conv18",
                    relu17, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn18   = batch_norm("bn18", conv18, is_training)
        relu18 = make_relu("relu18", bn18)

        # branch_pool
        pool19 = make_avg_pool_3x3_stride_1("pool19", intnsr, padding="SAME")
        conv19 = make_conv_1x1_no_bias("conv19",
                    pool19, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn19   = batch_norm("bn19", conv19, is_training)
        relu19 = make_relu("relu19", bn19)

        cc2    = make_concat("cc2", 3, [relu13, relu15, relu18, relu19])

    with tf.name_scope("mixed_35x35x288b"):
        intnsr = cc2

        # branch1x1
        conv20 = make_conv_1x1_no_bias("conv20",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn20   = batch_norm("bn20", conv20, is_training)
        relu20 = make_relu("relu20", bn20)

        # branch5x5
        conv21 = make_conv_1x1_no_bias("conv21",
                    intnsr, 48, padding="SAME", weight_decay=DECAY_RATE)
        bn21   = batch_norm("bn21", conv21, is_training)
        relu21 = make_relu("relu21", bn21)
        conv22 = make_conv_5x5_no_bias("conv22",
                    relu21, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn22   = batch_norm("bn22", conv22, is_training)
        relu22 = make_relu("relu22", bn22)

        # branch3x3dbl
        conv23 = make_conv_1x1_no_bias("conv23",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn23   = batch_norm("bn23", conv23, is_training)
        relu23 = make_relu("relu23", bn23)
        conv24 = make_conv_3x3_no_bias("conv24",
                    relu23, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn24   = batch_norm("bn24", conv24, is_training)
        relu24 = make_relu("relu24", bn24)
        conv25 = make_conv_3x3_no_bias("conv25",
                    relu24, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn25   = batch_norm("bn25", conv25, is_training)
        relu25 = make_relu("relu25", bn25)

        # branch_pool
        pool26 = make_avg_pool_3x3_stride_1("pool26", intnsr, padding="SAME")
        conv26 = make_conv_1x1_no_bias("conv26",
                    pool26, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn26   = batch_norm("bn26", conv26, is_training)
        relu26 = make_relu("relu26", bn26)

        cc3    = make_concat("cc3", 3, [relu20, relu22, relu25, relu26])

    with tf.name_scope("mixed_17x17x768a"):
        intnsr = cc3

        # branch3x3
        conv27 = make_conv_3x3_stride_2_no_bias("conv27",
                    intnsr, 384, weight_decay=DECAY_RATE)
        bn27   = batch_norm("bn27", conv27, is_training)
        relu27 = make_relu("relu27", bn27)

        # branch3x3dbl
        conv28 = make_conv_1x1_no_bias("conv28",
                    intnsr, 64, padding="SAME", weight_decay=DECAY_RATE)
        bn28   = batch_norm("bn28", conv28, is_training)
        relu28 = make_relu("relu28", bn28)
        conv29 = make_conv_3x3_no_bias("conv29",
                    relu28, 96, padding="SAME", weight_decay=DECAY_RATE)
        bn29   = batch_norm("bn29", conv29, is_training)
        relu29 = make_relu("relu29", bn29)
        conv30 = make_conv_3x3_stride_2_no_bias("conv30",
                    relu29, 96, weight_decay=DECAY_RATE)
        bn30   = batch_norm("bn30", conv30, is_training)
        relu30 = make_relu("relu30", bn30)

        pool31 = make_max_pool_3x3("pool31", intnsr)

        cc4    = make_concat("cc4", 3, [relu27, relu30, pool31])

    with tf.name_scope("mixed_17x17x768b"):
        intnsr = cc4

        # branch1x1
        conv32 = make_conv_1x1_no_bias("conv32",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn32   = batch_norm("bn32", conv32, is_training)
        relu32 = make_relu("relu32", bn32)

        # branch7x7
        conv33 = make_conv_1x1_no_bias("conv33",
                    intnsr, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn33   = batch_norm("bn33", conv33, is_training)
        relu33 = make_relu("relu33", bn33)
        conv34 = make_conv_1x7_no_bias("conv34",
                    relu33, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn34   = batch_norm("bn34", conv34, is_training)
        relu34 = make_relu("relu34", bn34)
        conv35 = make_conv_7x1_no_bias("conv35",
                    relu34, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn35   = batch_norm("bn35", conv35, is_training)
        relu35 = make_relu("relu35", bn35)

        # branch7x7dbl
        conv36 = make_conv_1x1_no_bias("conv36",
                    intnsr, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn36   = batch_norm("bn36", conv36, is_training)
        relu36 = make_relu("relu36", bn36)
        conv37 = make_conv_7x1_no_bias("conv37",
                    relu36, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn37   = batch_norm("bn37", conv37, is_training)
        relu37 = make_relu("relu37", bn37)
        conv38 = make_conv_1x7_no_bias("conv38",
                    relu37, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn38   = batch_norm("bn38", conv38, is_training)
        relu38 = make_relu("relu38", bn38)
        conv39 = make_conv_7x1_no_bias("conv39",
                    relu38, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn39   = batch_norm("bn39", conv39, is_training)
        relu39 = make_relu("relu39", bn39)
        conv40 = make_conv_1x7_no_bias("conv40",
                    relu39, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn40   = batch_norm("bn40", conv40, is_training)
        relu40 = make_relu("relu40", bn40)

        # branch_pool
        pool41 = make_avg_pool_3x3_stride_1("pool41", intnsr, padding="SAME")
        conv41 = make_conv_1x1_no_bias("conv41",
                    pool41, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn41   = batch_norm("bn41", conv41, is_training)
        relu41 = make_relu("relu41", bn41)

        cc5    = make_concat("cc5", 3, [relu32, relu35, relu40, relu41])

    with tf.name_scope("mixed_17x17x768c"):
        intnsr = cc5

        # branch1x1
        conv42 = make_conv_1x1_no_bias("conv42",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn42   = batch_norm("bn42", conv42, is_training)
        relu42 = make_relu("relu42", bn42)

        # branch7x7
        conv43 = make_conv_1x1_no_bias("conv43",
                    intnsr, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn43   = batch_norm("bn43", conv43, is_training)
        relu43 = make_relu("relu43", bn43)
        conv44 = make_conv_1x7_no_bias("conv44",
                    relu43, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn44   = batch_norm("bn44", conv44, is_training)
        relu44 = make_relu("relu44", bn44)
        conv45 = make_conv_7x1_no_bias("conv45",
                    relu44, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn45   = batch_norm("bn45", conv45, is_training)
        relu45 = make_relu("relu45", bn45)

        # branch7x7dbl
        conv46 = make_conv_1x1_no_bias("conv46",
                    intnsr, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn46   = batch_norm("bn46", conv46, is_training)
        relu46 = make_relu("relu46", bn46)
        conv47 = make_conv_7x1_no_bias("conv47",
                    relu46, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn47   = batch_norm("bn47", conv47, is_training)
        relu47 = make_relu("relu47", bn47)
        conv48 = make_conv_1x7_no_bias("conv48",
                    relu47, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn48   = batch_norm("bn48", conv48, is_training)
        relu48 = make_relu("relu48", bn48)
        conv49 = make_conv_7x1_no_bias("conv49",
                    relu48, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn49   = batch_norm("bn49", conv49, is_training)
        relu49 = make_relu("relu49", bn49)
        conv50 = make_conv_1x7_no_bias("conv50",
                    relu49, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn50   = batch_norm("bn50", conv50, is_training)
        relu50 = make_relu("relu50", bn50)

        # branch_pool
        pool51 = make_avg_pool_3x3_stride_1("pool51", intnsr, padding="SAME")
        conv51 = make_conv_1x1_no_bias("conv51",
                    pool51, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn51   = batch_norm("bn51", conv51, is_training)
        relu51 = make_relu("relu51", bn51)

        cc6    = make_concat("cc6", 3, [relu42, relu45, relu50, relu51])

    with tf.name_scope("mixed_17x17x768d"):
        intnsr = cc6

        # branch1x1
        conv52 = make_conv_1x1_no_bias("conv52",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn52   = batch_norm("bn52", conv52, is_training)
        relu52 = make_relu("relu52", bn52)

        # branch7x7
        conv53 = make_conv_1x1_no_bias("conv53",
                    intnsr, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn53   = batch_norm("bn53", conv53, is_training)
        relu53 = make_relu("relu53", bn53)
        conv54 = make_conv_1x7_no_bias("conv54",
                    relu53, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn54   = batch_norm("bn54", conv54, is_training)
        relu54 = make_relu("relu54", bn54)
        conv55 = make_conv_7x1_no_bias("conv55",
                    relu54, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn55   = batch_norm("bn55", conv55, is_training)
        relu55 = make_relu("relu55", bn55)

        # branch7x7dbl
        conv56 = make_conv_1x1_no_bias("conv56",
                    intnsr, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn56   = batch_norm("bn56", conv56, is_training)
        relu56 = make_relu("relu56", bn56)
        conv57 = make_conv_7x1_no_bias("conv57",
                    relu56, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn57   = batch_norm("bn57", conv57, is_training)
        relu57 = make_relu("relu57", bn57)
        conv58 = make_conv_1x7_no_bias("conv58",
                    relu57, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn58   = batch_norm("bn58", conv58, is_training)
        relu58 = make_relu("relu58", bn58)
        conv59 = make_conv_7x1_no_bias("conv59",
                    relu58, 160, padding="SAME", weight_decay=DECAY_RATE)
        bn59   = batch_norm("bn59", conv59, is_training)
        relu59 = make_relu("relu59", bn59)
        conv60 = make_conv_1x7_no_bias("conv60",
                    relu59, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn60   = batch_norm("bn60", conv60, is_training)
        relu60 = make_relu("relu60", bn60)

        # branch_pool
        pool61 = make_avg_pool_3x3_stride_1("pool61", intnsr, padding="SAME")
        conv61 = make_conv_1x1_no_bias("conv61",
                    pool61, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn61   = batch_norm("bn61", conv61, is_training)
        relu61 = make_relu("relu61", bn61)

        cc7    = make_concat("cc7", 3, [relu52, relu55, relu60, relu61])

    with tf.name_scope("mixed_17x17x768e"):
        intnsr = cc7

        # branch1x1
        conv62 = make_conv_1x1_no_bias("conv62",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn62   = batch_norm("bn62", conv62, is_training)
        relu62 = make_relu("relu62", bn62)

        # branch7x7
        conv63 = make_conv_1x1_no_bias("conv63",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn63   = batch_norm("bn63", conv63, is_training)
        relu63 = make_relu("relu63", bn63)
        conv64 = make_conv_1x7_no_bias("conv64",
                    relu63, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn64   = batch_norm("bn64", conv64, is_training)
        relu64 = make_relu("relu64", bn64)
        conv65 = make_conv_7x1_no_bias("conv65",
                    relu64, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn65   = batch_norm("bn65", conv65, is_training)
        relu65 = make_relu("relu65", bn65)

        # branch7x7dbl
        conv66 = make_conv_1x1_no_bias("conv66",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn66   = batch_norm("bn66", conv66, is_training)
        relu66 = make_relu("relu66", bn66)
        conv67 = make_conv_7x1_no_bias("conv67",
                    relu66, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn67   = batch_norm("bn67", conv67, is_training)
        relu67 = make_relu("relu67", bn67)
        conv68 = make_conv_1x7_no_bias("conv68",
                    relu67, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn68   = batch_norm("bn68", conv68, is_training)
        relu68 = make_relu("relu68", bn68)
        conv69 = make_conv_7x1_no_bias("conv69",
                    relu68, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn69   = batch_norm("bn69", conv69, is_training)
        relu69 = make_relu("relu69", bn69)
        conv70 = make_conv_1x7_no_bias("conv70",
                    relu69, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn70   = batch_norm("bn70", conv70, is_training)
        relu70 = make_relu("relu70", bn70)

        # branch_pool
        pool71 = make_avg_pool_3x3_stride_1("pool71", intnsr, padding="SAME")
        conv71 = make_conv_1x1_no_bias("conv71",
                    pool71, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn71   = batch_norm("bn71", conv71, is_training)
        relu71 = make_relu("relu71", bn71)

        cc8    = make_concat("cc8", 3, [relu62, relu65, relu70, relu71])

    with tf.name_scope("aux_logits"):
        intnsr = cc8

        # aux_logits
        pool72 = make_avg_pool_5x5_stride_3("pool72", intnsr)
        conv72 = make_conv_1x1_no_bias("conv72",
                    pool72, 128, padding="SAME", weight_decay=DECAY_RATE)
        bn72   = batch_norm("bn72", conv72, is_training)
        relu72 = make_relu("relu72", bn72)

        shape  = relu72.get_shape()
        conv73 = make_conv_no_bias("conv73",
                    relu72, shape[1], shape[2], 128,
                    weight_decay=DECAY_RATE, stddev=0.01)
        bn73   = batch_norm("bn73", conv73, is_training)
        relu73 = make_relu("relu73", bn73)

        aux_flat   = make_flatten("aux_flatten", relu73)
        aux_logits = make_fc("aux_fc",
                        aux_flat, count_classes,
                        weight_decay=DECAY_RATE, stddev=0.001,)

    with tf.name_scope("mixed_8x8x1280a"):
        intnsr = cc8

        # branch3x3
        conv74 = make_conv_1x1_no_bias("conv74",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn74   = batch_norm("bn74", conv74, is_training)
        relu74 = make_relu("relu74", bn74)
        conv75 = make_conv_3x3_stride_2_no_bias("conv75",
                    relu74, 320, weight_decay=DECAY_RATE)
        bn75   = batch_norm("bn75", conv75, is_training)
        relu75 = make_relu("relu75", bn75)

        # branch7x7x3
        conv76 = make_conv_1x1_no_bias("conv76",
                    intnsr, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn76   = batch_norm("bn76", conv76, is_training)
        relu76 = make_relu("relu76", bn76)
        conv77 = make_conv_1x7_no_bias("conv77",
                    relu76, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn77   = batch_norm("bn77", conv77, is_training)
        relu77 = make_relu("relu77", bn77)
        conv78 = make_conv_7x1_no_bias("conv78",
                    relu77, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn78   = batch_norm("bn78", conv78, is_training)
        relu78 = make_relu("relu78", bn78)
        conv79 = make_conv_3x3_stride_2_no_bias("conv79",
                    relu78, 192, weight_decay=DECAY_RATE)
        bn79   = batch_norm("bn79", conv79, is_training)
        relu79 = make_relu("relu79", bn79)

        # branch_pool
        pool80 = make_max_pool_3x3("pool80", intnsr)

        cc9    = make_concat("cc9", 3, [relu75, relu79, pool80])

    with tf.name_scope("mixed_8x8x2048a"):
        intnsr = cc9

        # branch1x1
        conv81 = make_conv_1x1_no_bias("conv81",
                    intnsr, 320, padding="SAME", weight_decay=DECAY_RATE)
        bn81   = batch_norm("bn81", conv81, is_training)
        relu81 = make_relu("relu81", bn81)

        # branch3x3
        conv82 = make_conv_1x1_no_bias("conv82",
                    intnsr, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn82   = batch_norm("bn82", conv82, is_training)
        relu82 = make_relu("relu82", bn82)
        conv83 = make_conv_1x3_no_bias("conv83",
                    relu82, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn83   = batch_norm("bn83", conv83, is_training)
        relu83 = make_relu("relu83", bn83)
        conv84 = make_conv_3x1_no_bias("conv84",
                    relu82, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn84   = batch_norm("bn84", conv84, is_training)
        relu84 = make_relu("relu84", bn84)
        cc10   = make_concat("cc10", 3, [relu83, relu84])

        # branch3x3dbl
        conv85 = make_conv_1x1_no_bias("conv85",
                    intnsr, 448, padding="SAME", weight_decay=DECAY_RATE)
        bn85   = batch_norm("bn85", conv85, is_training)
        relu85 = make_relu("relu85", bn85)
        conv86 = make_conv_3x3_no_bias("conv86",
                    relu85, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn86   = batch_norm("bn86", conv86, is_training)
        relu86 = make_relu("relu86", bn86)
        conv87 = make_conv_1x3_no_bias("conv87",
                    relu86, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn87   = batch_norm("bn87", conv87, is_training)
        relu87 = make_relu("relu87", bn87)
        conv88 = make_conv_3x1_no_bias("conv88",
                    relu86, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn88   = batch_norm("bn88", conv88, is_training)
        relu88 = make_relu("relu88", bn88)
        cc11   = make_concat("cc11", 3, [relu87, relu88])

        # branch_pool
        pool89 = make_avg_pool_3x3_stride_1("pool89", intnsr, padding="SAME")
        conv89 = make_conv_1x1_no_bias("conv89",
                    pool89, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn89   = batch_norm("bn89", conv89, is_training)
        relu89 = make_relu("relu89", bn89)

        cc12   = make_concat("cc12", 3, [relu81, cc10, cc11, relu89])

    with tf.name_scope("mixed_8x8x2048b"):
        intnsr = cc12

        # branch1x1
        conv90 = make_conv_1x1_no_bias("conv90",
                    intnsr, 320, padding="SAME", weight_decay=DECAY_RATE)
        bn90   = batch_norm("bn90", conv90, is_training)
        relu90 = make_relu("relu90", bn90)

        # branch3x3
        conv91 = make_conv_1x1_no_bias("conv91",
                    intnsr, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn91   = batch_norm("bn91", conv91, is_training)
        relu91 = make_relu("relu91", bn91)
        conv92 = make_conv_1x3_no_bias("conv92",
                    relu91, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn92   = batch_norm("bn92", conv92, is_training)
        relu92 = make_relu("relu92", bn92)
        conv93 = make_conv_3x1_no_bias("conv93",
                    relu91, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn93   = batch_norm("bn93", conv93, is_training)
        relu93 = make_relu("relu93", bn93)
        cc13   = make_concat("cc13", 3, [relu92, relu93])

        # branch3x3dbl
        conv94 = make_conv_1x1_no_bias("conv94",
                    intnsr, 448, padding="SAME", weight_decay=DECAY_RATE)
        bn94   = batch_norm("bn94", conv94, is_training)
        relu94 = make_relu("relu94", bn94)
        conv95 = make_conv_3x3_no_bias("conv95",
                    relu94, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn95   = batch_norm("bn95", conv95, is_training)
        relu95 = make_relu("relu95", bn95)
        conv96 = make_conv_1x3_no_bias("conv96",
                    relu95, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn96   = batch_norm("bn96", conv96, is_training)
        relu96 = make_relu("relu96", bn96)
        conv97 = make_conv_3x1_no_bias("conv97",
                    relu95, 384, padding="SAME", weight_decay=DECAY_RATE)
        bn97   = batch_norm("bn97", conv97, is_training)
        relu97 = make_relu("relu97", bn97)
        cc14   = make_concat("cc14", 3, [relu96, relu97])

        # branch_pool
        pool98 = make_avg_pool_3x3_stride_1("pool98", intnsr, padding="SAME")
        conv98 = make_conv_1x1_no_bias("conv98",
                    pool98, 192, padding="SAME", weight_decay=DECAY_RATE)
        bn98   = batch_norm("bn98", conv98, is_training)
        relu98 = make_relu("relu98", bn98)

        cc15   = make_concat("cc15", 3, [relu90, cc13, cc14, relu98])

    with tf.name_scope("logits"):
        intnsr = cc15

        shape  = intnsr.get_shape()
        pool99 = make_avg_pool("pool99", intnsr, shape[1], shape[2])

        keep_prob = tf.cond(is_training,
                            lambda: tf.constant(0.8),
                            lambda: tf.constant(1.0),
                            name="keep_prob")

        do99   = make_dropout("do99", pool99, keep_prob)

        flat   = make_flatten("flatten", do99)
        logits = make_fc("fc", flat, count_classes, weight_decay=DECAY_RATE)

    with tf.name_scope("loss"):
        y_out     = tf.stop_gradient(y_out)
        aux_preds = lsr_loss(aux_logits, y_out, 0.1, 0.4)
        preds     = lsr_loss(logits, y_out, 0.1, 1.0)
        loss      = tf.reduce_mean(aux_preds+preds)
    return logits, loss


def run_towers(optimizer, global_step, is_training, is_nbl,
        training_data, validation_data, nbl_val_data, count_classes, num_gpus):
    with tf.device("/device:CPU:0"), tf.name_scope("input/train_or_eval"):
        images, labels = \
            tf.cond(is_training, lambda: training_data, lambda:
            tf.cond(is_nbl, lambda: nbl_val_data, lambda: validation_data))
    labels_list = []
    logits_list = []
    loss_list   = []
    grads       = []
    for i in range(num_gpus):
        tower_name = "tower%d" % i
        with tf.device("/device:GPU:%d" % i):
            with tf.name_scope(tower_name):
                logits, loss = make_tower(
                    images, labels, is_training, count_classes)
            labels_list.append(labels)
            logits_list.append(logits)
            loss_list.append(loss)
            grads.append(optimizer.compute_gradients(loss))

    train_op, loss_op, acc_top_1, acc_top_5 = merge_towers_and_optimize(
        optimizer, global_step, grads, logits_list, loss_list, labels_list)

    return train_op, loss_op, acc_top_1, acc_top_5
