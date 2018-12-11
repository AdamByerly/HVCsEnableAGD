import argparse
import tensorflow as tf
from datetime import datetime
from output import Output
from imagenet.input_sieve import DataSet, train_inputs, eval_inputs
from inception_v3.model import run_towers, apply_gradients
from inception_v3.model import compute_total_loss, evaluate_validation


def go(start_epoch, end_epoch, run_name, weights_file,
       profile_compute_time_every_n_steps, save_summary_info_every_n_steps,
       log_annotated_images):
    tf.reset_default_graph()

    out = Output(run_name, profile_compute_time_every_n_steps,
                 save_summary_info_every_n_steps)

    out.log_msg("Setting up data feeds...")
    training_dataset   = DataSet('train', 128)
    validation_dataset = DataSet('validation', 128)
    training_data      = train_inputs(training_dataset,
                            64, 299, 299, log_annotated_images)
    validation_data    = eval_inputs(validation_dataset,
                            64, 299, 299, log_annotated_images)
    training_steps     = training_dataset.num_batches_per_epoch()
    validation_steps   = validation_dataset.num_batches_per_epoch()

    with tf.device("/device:CPU:0"):
        global_step = tf.train.get_or_create_global_step()
        with tf.name_scope("input/placeholders"):
            keep_prob   = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool)

    opt = tf.train.AdamOptimizer()

    loss1, loss2, logits, labels = run_towers(keep_prob,
        is_training, training_data, validation_data, DataSet.num_classes())
    train_op = apply_gradients(loss1, loss2, global_step, opt)
    loss = compute_total_loss(loss1, loss2)
    acc_top_1, acc_top_5 = evaluate_validation(logits, labels)

    out.log_msg("Starting Session...")
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        out.set_session_graph(sess.graph)

        if weights_file is not None:
            out.log_msg("Restoring weights file: {}".format(weights_file))
            tf.train.Saver().restore(sess, weights_file)
        else:
            tf.global_variables_initializer().run()

        tf.train.start_queue_runners(sess=sess)

        for e in range(start_epoch, end_epoch+1):
            for i in range(training_steps):
                out.train_step_begin(i)

                _, l, g_step = sess.run([train_op, loss, global_step],
                    feed_dict={keep_prob: 0.8, is_training: True},
                    options=out.get_run_options(),
                    run_metadata=out.get_run_metadata())

                out.train_step_end(sess, g_step, e, i, l, training_steps,
                    feed_dict={keep_prob: 0.8, is_training: True})

            acc_top1, acc_top5, test_loss = (0, 0, 0)
            for i in range(validation_steps):
                out.validation_step_begin(i, validation_steps)

                l, acc1, acc5 = sess.run([loss, acc_top_1, acc_top_5],
                    feed_dict={keep_prob: 1, is_training: False})
                acc_top1  = (acc1+(i*acc_top1))/(i+1)
                acc_top5  = (acc5+(i*acc_top5))/(i+1)
                test_loss = (l+(i*test_loss))/(i+1)

            out.end_epoch(sess, e, global_step, test_loss, acc_top1, acc_top5)

    out.close_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inception_v3")
    parser.add_argument("-se", "--start_epoch", default=1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=100, type=int)
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
