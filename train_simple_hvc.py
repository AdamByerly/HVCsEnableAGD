import argparse
import tensorflow as tf
from datetime import datetime
from cnn_helpers import apply_gradients, compute_total_loss, evaluate_validation
from simple.output import Output
from simple.input_sieve import DataSet, train_inputs, eval_inputs
from simple.model_hvc import run_towers


def train(out, sess, epoch, training_steps, train_op, loss_op,
          global_step, is_training_ph):
    g_step = 0
    for i in range(training_steps):
        out.train_step_begin(i)

        _, l, g_step = sess.run(
            [train_op, loss_op, global_step],
            feed_dict={is_training_ph: True},
            options=out.get_run_options(),
            run_metadata=out.get_run_metadata())

        out.train_step_end(
            sess, epoch, g_step, i, l, "DEFAULT", training_steps,
            feed_dict={is_training_ph: True})

    out.train_end(sess, epoch, g_step)


def validate(out, sess, epoch, validation_steps,
             loss_op, acc_top_1_op, acc_top_5_op, global_step,
             is_training_ph):
    g_step, acc_top1, acc_top5, test_loss = (0, 0, 0, 0)
    for i in range(validation_steps):
        out.validation_step_begin(i, validation_steps)

        g_step, l, acc1, acc5 = sess.run(
            [global_step, loss_op, acc_top_1_op, acc_top_5_op],
            feed_dict={is_training_ph: False})
        acc_top1 = (acc1 + (i * acc_top1)) / (i + 1)
        acc_top5 = (acc5 + (i * acc_top5)) / (i + 1)
        test_loss = (l + (i * test_loss)) / (i + 1)

    out.validation_end(sess, epoch, g_step, False,
                       test_loss, "DEFAULT", acc_top1, acc_top5)


def go(start_epoch, end_epoch, run_name, weights_file,
       profile_compute_time_every_n_steps, save_summary_info_every_n_steps,
       log_annotated_images):
    tf.reset_default_graph()

    out = Output(run_name, profile_compute_time_every_n_steps,
                 save_summary_info_every_n_steps)

    ############################################################################
    # Data feeds
    ############################################################################
    out.log_msg("Setting up data feeds...")
    training_dataset   = DataSet('train')
    validation_dataset = DataSet('validation')
    training_data      = train_inputs(training_dataset, log_annotated_images)
    validation_data    = eval_inputs(validation_dataset, log_annotated_images)
    training_steps     = training_dataset.num_batches_per_epoch()
    validation_steps   = validation_dataset.num_batches_per_epoch()

    ############################################################################
    # Tensorflow placeholders and operations
    ############################################################################
    with tf.device("/device:CPU:0"):  # set the default device to the CPU
        with tf.name_scope("input/placeholders"):
            is_training_ph     = tf.placeholder(tf.bool)

            global_step        = tf.train.get_or_create_global_step()
            opt                = tf.train.AdamOptimizer()

            loss1, loss2, \
                logits, labels = run_towers(is_training_ph,
                                    training_data, validation_data,
                                    DataSet.num_classes())
            train_op           = apply_gradients(loss1, loss2, global_step, opt)
            loss_op            = compute_total_loss(loss1, loss2)
            acc_top_1_op, \
                acc_top_5_op   = evaluate_validation(logits, labels)

    out.log_msg("Starting Session...")

    ############################################################################
    # Tensorflow session
    ############################################################################
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
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
                      global_step, is_training_ph)

                validate(out, sess, e, validation_steps, loss_op, acc_top_1_op,
                         acc_top_5_op, global_step, is_training_ph)
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
    parser = argparse.ArgumentParser(description="ImageNetNet")
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
    args = parser.parse_args()
    print(args)

    go(args.start_epoch, args.end_epoch, args.run_name, args.weights_file,
       args.profile_compute_time_every_n_steps,
       args.save_summary_info_every_n_steps, args.log_annotated_images)
