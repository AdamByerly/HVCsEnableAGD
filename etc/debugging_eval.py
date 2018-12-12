import os
import tensorflow as tf
from imagenet.output import Output
from imagenet.input_sieve import DataSet, eval_inputs
from simple.model import run_towers
from simple.model import compute_total_loss, evaluate_validation


def go(run_name, weights_file, log_annotated_images):
    tf.reset_default_graph()

    out = Output(run_name)

    out.log_msg("Setting up data feeds...")
    validation_dataset = DataSet('validation', 128)
    validation_data = eval_inputs(validation_dataset,
                        128, 224, 224, log_annotated_images)
    validation_steps = validation_dataset.num_batches_per_epoch()

    with tf.device("/device:CPU:0"):
        with tf.name_scope("input/placeholders"):
            keep_prob = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool)

    loss1, loss2, logits, labels = run_towers(keep_prob,
        is_training, validation_data, validation_data, DataSet.num_classes())
    loss = compute_total_loss(loss1, loss2)
    acc_top_1, acc_top_5 = evaluate_validation(logits, labels)

    out.log_msg("Starting Session...")
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        out.set_session_graph(sess.graph)

        tb_writer = tf.summary.FileWriter(
            os.path.join("logs", run_name), sess.graph)

        if weights_file is not None:
            out.log_msg("Restoring weights file: {}".format(weights_file))
            tf.train.Saver().restore(sess, weights_file)
        else:
            tf.global_variables_initializer().run()

        tf.train.start_queue_runners(sess=sess)

        acc_top1, acc_top5, test_loss = (0, 0, 0)
        for i in range(validation_steps):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            out.log_msg("Validating (step {}/{})...".
                        format(i + 1, validation_steps + 1), True)
            l, acc1, acc5 = sess.run([loss, acc_top_1, acc_top_5, logits],
                    feed_dict={keep_prob: 1, is_training: False},
                    options=run_options, run_metadata=run_metadata)
            acc_top1 = (acc1 + (i * acc_top1)) / (i + 1)
            acc_top5 = (acc5 + (i * acc_top5)) / (i + 1)
            test_loss = (l + (i * test_loss)) / (i + 1)

            out.log_run_metadata(run_metadata, i)
            summary_op = tf.summary.merge_all()
            summary_str = sess.run(summary_op)
            tb_writer.add_summary(summary_str)
            tb_writer.flush()

            out.log_msg("Loss : {}".format(test_loss), False)
            out.log_msg("Top-1: {}".format(acc_top1), False)
            out.log_msg("Top-1: {}".format(acc_top5), False)


if __name__ == "__main__":
    go("DEBUG_EVAL_01", "./logs/20181113123058/weights-1-latest-10010", True)
