import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from MNIST.IO import IO
from MNIST.model import run_towers, apply_gradients
from MNIST.model import compute_total_loss, evaluate_validation

BATCH_SIZE = 128


def shift_2(batch):
    random_nums = (np.random.rand(batch.shape[0]*2)*5).astype(int)-2
    for i in range(batch.shape[0]):
        new_image      = np.zeros((batch.shape[1],
                            batch.shape[2], batch.shape[3]))
        nonzero_x_cols = np.argwhere(batch[i].sum(axis=0) > 0)[:, 0]
        nonzero_y_rows = np.argwhere(batch[i].sum(axis=1) > 0)[:, 0]
        left_margin    = np.min(nonzero_x_cols)
        right_margin   = batch.shape[1] - np.max(nonzero_x_cols) - 1
        top_margin     = np.min(nonzero_y_rows)
        bot_margin     = batch.shape[2] - np.max(nonzero_y_rows) - 1
        shift_x        = random_nums[i*2]
        shift_y        = random_nums[i*2+1]
        shift_x        = (np.maximum(shift_x, -left_margin)) \
                            if shift_x < 0 else \
                                np.minimum(shift_x, right_margin)
        shift_y        = (np.maximum(shift_y, -top_margin)) \
                            if shift_y < 0 else \
                                np.minimum(shift_y, bot_margin)
        src_x_start    = -1 * np.minimum(0, shift_x)
        src_y_start    = -1 * np.minimum(0, shift_y)
        dest_x_start   = np.maximum(0, shift_x)
        dest_y_start   = np.maximum(0, shift_y)
        src_x_end      = batch.shape[1]-dest_x_start
        src_y_end      = batch.shape[2]-dest_y_start
        dest_x_end     = batch.shape[1]-src_x_start
        dest_y_end     = batch.shape[2]-src_y_start
        new_image[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
            batch[i, src_y_start:src_y_end, src_x_start:src_x_end]
        batch[i]       = new_image
    return batch


def go(start_epoch, end_epoch, run_name, weights_file,
       profile_compute_time_every_n_steps, save_summary_info_every_n_steps):
    tf.reset_default_graph()

    io = IO(run_name, profile_compute_time_every_n_steps,
            save_summary_info_every_n_steps)

    io.log_msg("Setting up data feeds...")
    io.log_msg("Loading MNIST training data...")
    train_x, train_y = io.load_mnist_train()
    io.log_msg("MNIST training data loaded.")
    io.log_msg("Loading MNIST testing data...")
    test_x, test_y   = io.load_mnist_test()
    io.log_msg("MNIST testing data loaded.")
    training_steps   = io.num_training_batches_per_epoch()
    validation_steps = io.num_validation_batches()

    with tf.device("/device:CPU:0"):
        global_step = tf.train.get_or_create_global_step()
        with tf.name_scope("input/placeholders"):
            x_in        = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            y_out       = tf.placeholder(tf.float32, shape=[None, 10])
            is_training = tf.placeholder(tf.bool)

    with tf.device("/device:CPU:0"), tf.name_scope("input/"):
        ds       = tf.data.Dataset.from_tensor_slices((x_in, y_out))
        ds       = ds.batch(BATCH_SIZE)
        iterator = ds.make_initializable_iterator()

    opt            = tf.train.AdamOptimizer()
    loss_labels1,\
        loss_labels2,\
        loss_recon1,\
        loss_recon2,\
        loss_total1,\
        loss_total2,\
        logits,\
        labels     = run_towers(iterator, is_training)
    train_op       = apply_gradients(loss_total1, loss_total2, global_step, opt)
    loss_labels,\
        loss_recon, \
        loss_total = compute_total_loss(loss_labels1, loss_labels2,
                        loss_recon1, loss_recon2, loss_total1, loss_total2)
    accuracy       = evaluate_validation(logits, labels)

    io.log_msg("Starting Session...")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        io.set_session_graph(sess.graph)

        if weights_file is not None:
            io.log_msg("Restoring weights file: {}".format(weights_file))
            tf.train.Saver().restore(sess, weights_file)
        else:
            tf.global_variables_initializer().run()

        for e in range(start_epoch, end_epoch+1):
            all_indices = np.arange(train_x.shape[0])
            index_perms = np.random.permutation(all_indices.shape[0])
            train_x     = train_x[index_perms]
            # train_x     = shift_2(np.copy(train_x))
            train_y     = train_y[index_perms]

            sess.run(iterator.initializer,
                feed_dict={x_in: train_x, y_out: train_y})

            for i in range(training_steps):
                io.train_step_begin(i)

                _, ll, lr, lt, g_step = sess.run([train_op,
                    loss_labels, loss_recon, loss_total, global_step],
                    feed_dict={is_training: True},
                    options=io.get_run_options(),
                    run_metadata=io.get_run_metadata())

                io.train_step_end(sess, g_step,
                    e, i, ll, lr, lt, feed_dict={is_training: True})

            sess.run(iterator.initializer,
                feed_dict={x_in: test_x, y_out: test_y})

            test_acc, test_loss_labels,\
                test_loss_recon, test_loss_total = (0, 0, 0, 0)
            for i in range(validation_steps):
                io.validation_step_begin(i, validation_steps)

                ll, lr, lt, acc = sess.run([loss_labels, loss_recon,
                    loss_total, accuracy], feed_dict={is_training: False})
                test_acc  = (acc+(i*test_acc))/(i+1)
                test_loss_labels = (ll+(i*test_loss_labels))/(i+1)
                test_loss_recon  = (lr+(i*test_loss_recon))/(i+1)
                test_loss_total  = (lt+(i*test_loss_total))/(i+1)
                io.end_epoch(sess, e, global_step, test_loss_labels,
                    test_loss_recon, test_loss_total, test_acc)

    io.close_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNetNet")
    parser.add_argument("-se", "--start_epoch", default=1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=50, type=int)
    parser.add_argument("-rn", "--run_name",
        default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument("-wf", "--weights_file", default=None)
    parser.add_argument("-pct", "--profile_compute_time_every_n_steps",
        default=None, type=int)
    parser.add_argument("-ssi", "--save_summary_info_every_n_steps",
        default=None, type=int)
    args = parser.parse_args()
    print(args)

    go(args.start_epoch, args.end_epoch, args.run_name,
       args.weights_file, args.profile_compute_time_every_n_steps,
       args.save_summary_info_every_n_steps)
