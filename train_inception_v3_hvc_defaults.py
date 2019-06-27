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

import argparse
import tensorflow as tf
from datetime import datetime
from input_sieve import DataSet, train_inputs, eval_inputs
from input_sieve import non_blacklisted_eval_inputs
from inception_v3.output import Output
from inception_v3.model_hvc import run_towers


def train(out, sess, epoch, training_steps, train_op, loss_op,
        global_step, is_training_ph, is_validating_nbl_ph):
    g_step = 0
    for i in range(training_steps):
        out.train_step_begin(i)

        _, l, g_step = sess.run(
            [train_op, loss_op, global_step],
            feed_dict={is_training_ph: True,
                       is_validating_nbl_ph: False},
            options=out.get_run_options(),
            run_metadata=out.get_run_metadata())

        out.train_step_end(
            sess, epoch, g_step, i, l, "DEFAULT", training_steps,
            feed_dict={is_training_ph: True,
                       is_validating_nbl_ph: False})

    out.train_end(sess, epoch, g_step)


def validate(out, sess, epoch, validation_steps, loss_op,
        acc_top_1_op, acc_top_5_op, global_step,
        is_training_ph, is_validating_nbl_ph):
    g_step, acc_top1, acc_top5, test_loss = (0, 0, 0, 0)
    for i in range(validation_steps):
        out.validation_step_begin(i, validation_steps)

        g_step, l, acc1, acc5 = sess.run(
            [global_step, loss_op, acc_top_1_op, acc_top_5_op],
            feed_dict={is_training_ph: False,
                       is_validating_nbl_ph: False})
        acc_top1 = (acc1 + (i * acc_top1)) / (i + 1)
        acc_top5 = (acc5 + (i * acc_top5)) / (i + 1)
        test_loss = (l + (i * test_loss)) / (i + 1)

    out.validation_end(sess, epoch, g_step, False,
                       test_loss, "DEFAULT", acc_top1, acc_top5)


def validate_nbl(out, sess, epoch, nbl_validation_steps, loss_op,
        acc_top_1_op, acc_top_5_op, global_step,
        is_training_ph, is_validating_nbl_ph):
    g_step, acc_top1, acc_top5, test_loss = (0, 0, 0, 0)
    for i in range(nbl_validation_steps):
        out.validation_step_begin(i, nbl_validation_steps)

        g_step, l, acc1, acc5 = sess.run(
            [global_step, loss_op, acc_top_1_op, acc_top_5_op],
            feed_dict={is_training_ph: False,
                       is_validating_nbl_ph: True})
        acc_top1 = (acc1 + (i * acc_top1)) / (i + 1)
        acc_top5 = (acc5 + (i * acc_top5)) / (i + 1)
        test_loss = (l + (i * test_loss)) / (i + 1)

    out.validation_end(sess, epoch, g_step, True,
                       test_loss, "DEFAULT", acc_top1, acc_top5)


def go(start_epoch, end_epoch, run_name, weights_file,
       profile_compute_time_every_n_steps, save_summary_info_every_n_steps,
       log_annotated_images, image_size, batch_size, num_gpus,
       data_dir, black_list_file, log_dir, do_validate_all, do_validate_nbl):
    tf.reset_default_graph()

    out = Output(log_dir, run_name, profile_compute_time_every_n_steps,
                 save_summary_info_every_n_steps)

    ############################################################################
    # Data feeds
    ############################################################################
    out.log_msg("Setting up data feeds...")
    training_dataset   = DataSet('train', image_size, batch_size, num_gpus,
                            data_dir, None)
    validation_dataset = DataSet('validation', image_size, batch_size, num_gpus,
                            data_dir, black_list_file)
    training_data      = train_inputs(training_dataset, log_annotated_images)
    validation_data    = eval_inputs(validation_dataset, log_annotated_images)
    nbl_val_data       = non_blacklisted_eval_inputs(
                            validation_dataset, log_annotated_images)
    training_steps     = training_dataset.training_batches_per_epoch()
    validation_steps   = validation_dataset.validation_batches_per_epoch()
    nbl_val_steps      = validation_dataset.nbl_validation_batches_per_epoch()

    ############################################################################
    # Tensorflow placeholders and operations
    ############################################################################
    with tf.device("/device:CPU:0"):  # set the default device to the CPU
        with tf.name_scope("input/placeholders"):
            is_training_ph       = tf.placeholder(tf.bool)
            is_validating_nbl_ph = tf.placeholder(tf.bool)

        global_step        = tf.train.get_or_create_global_step()
        opt                = tf.train.AdamOptimizer()

        train_op, loss_op,\
            acc_top_1_op, \
            acc_top_5_op   = run_towers(opt, global_step, is_training_ph,
                                        is_validating_nbl_ph, training_data,
                                        validation_data, nbl_val_data,
                                        DataSet.num_classes(), num_gpus)

    out.log_msg("Starting Session...")

    ############################################################################
    # Tensorflow session
    ############################################################################
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        out.set_session_graph(sess.graph)

        if weights_file is not None:
            out.log_msg("Restoring weights file: {}".format(weights_file))
            tf.train.Saver().restore(sess, weights_file)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for e in range(start_epoch, end_epoch + 1):
                train(out, sess, e, training_steps, train_op, loss_op,
                      global_step, is_training_ph, is_validating_nbl_ph)

                if do_validate_all:
                    validate(out, sess, e, validation_steps, loss_op,
                             acc_top_1_op, acc_top_5_op, global_step,
                             is_training_ph, is_validating_nbl_ph)

                if do_validate_nbl:
                    validate_nbl(out, sess, e, nbl_val_steps, loss_op,
                                 acc_top_1_op, acc_top_5_op, global_step,
                                 is_training_ph, is_validating_nbl_ph)
        except tf.errors.OutOfRangeError:
            out.log_msg("Finished.")
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    out.close_files()


################################################################################
# Entry point
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inception_v3")
    parser.add_argument("-se", "--start_epoch", default=1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=175, type=int)
    parser.add_argument("-rn", "--run_name",
        default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument("-wf", "--weights_file", default=None)
    parser.add_argument("-pct", "--profile_compute_time_every_n_steps",
        default=None, type=int)
    parser.add_argument("-ssi", "--save_summary_info_every_n_steps",
        default=None, type=int)
    parser.add_argument("-lai", "--log_annotated_images",
        default=False, type=bool)
    parser.add_argument("-is", "--image_size", default=299, type=int)
    parser.add_argument("-bs", "--batch_size", default=96, type=int)
    parser.add_argument("-g", "--gpus", default=2, type=int)
    parser.add_argument("-ld", "--log_dir", default="logs")
    parser.add_argument("-dd", "--data_dir",
        default="D:\\workingbasedir\\processed")
    parser.add_argument("-blf", "--black_list_file",
        default="ILSVRC2015_clsloc_validation_blacklist.txt")
    parser.add_argument("-vall", "--validate_all", default=True, type=bool)
    parser.add_argument("-vnbl", "--validate_nbl", default=True, type=bool)
    args = parser.parse_args()
    print(args)

    go(args.start_epoch, args.end_epoch, args.run_name, args.weights_file,
       args.profile_compute_time_every_n_steps,
       args.save_summary_info_every_n_steps, args.log_annotated_images,
       args.image_size, args.batch_size, args.gpus,
       args.data_dir, args.black_list_file, args.log_dir,
       args.validate_all, args.validate_nbl)
