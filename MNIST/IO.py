import os
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime

BATCH_SIZE    = 128
LOGS_BASE_DIR = "C:\\Users\\abyerly\\Desktop\\"\
                + "Machine Learning\\My Learning\\logs\\"
DATA_BASE_DIR = "C:\\Users\\abyerly\\Desktop\\" \
                + "Machine Learning\\My Learning\\mnist_data\\"


# noinspection PyPep8Naming
class IO:
    MNIST_TRAINING_IMAGE_COUNT = 60000
    MNIST_TESTING_IMAGE_COUNT = 10000
    CSV_FIELDS = ["Date", "Time", "Epoch", "Step", "Accuracy",
                  "Loss(Labels)", "Loss(Reconstruction)", "Loss(Total)"]

    def __init__(self, run_name, profile_compute_time_every_n_steps=None,
                 save_summary_info_every_n_steps=None):
        self.run_name = run_name
        self.profile_compute_time_every_n_steps\
            = profile_compute_time_every_n_steps
        self.save_summary_info_every_n_steps\
            = save_summary_info_every_n_steps
        self.model_file_base = os.path.join(
            os.path.join(LOGS_BASE_DIR, run_name), "weights")
        self.train_csv = self.open_csv(
            os.path.join(LOGS_BASE_DIR, "{}_training_log.csv".format(run_name)))
        self.test_csv = self.open_csv(
            os.path.join(LOGS_BASE_DIR, "{}_test_log.csv".format(run_name)))
        no_accuracy = self.CSV_FIELDS.copy()
        no_accuracy.remove("Accuracy")
        self.train_csv_writer = csv.DictWriter(self.train_csv,
            fieldnames=no_accuracy)
        no_step = self.CSV_FIELDS.copy()
        no_step.remove("Step")
        self.test_csv_writer = csv.DictWriter(self.test_csv,
            fieldnames=no_step)
        self.train_csv_writer.writeheader()
        self.test_csv_writer.writeheader()
        self.train_csv.flush()
        self.test_csv.flush()
        self.tb_writer = None
        self.tf_saver_best = None
        self.tf_saver_latest = None
        self.run_options = None
        self.run_metadata = None
        self.best_accuracy = 0

    def get_run_options(self):
        return self.run_options

    def get_run_metadata(self):
        return self.run_metadata

    def set_session_graph(self, session_graph):
        self.tb_writer = tf.summary.FileWriter(
            os.path.join(LOGS_BASE_DIR, self.run_name), session_graph)
        self.tf_saver_best = tf.train.Saver(max_to_keep=5)
        self.tf_saver_latest = tf.train.Saver(max_to_keep=5)

    def close_files(self):
        self.train_csv.close()
        self.test_csv.close()
        self.tb_writer.close()

    def save_model_best(self, session, epoch, g_step):
        self.tf_saver_best.save(session, "{}-{}-best"
            .format(self.model_file_base, epoch), global_step=g_step)

    def save_model_latest(self, session, epoch, g_step):
        self.tf_saver_latest.save(session, "{}-{}-latest"
            .format(self.model_file_base, epoch), global_step=g_step)

    def train_step_begin(self, step):
        if self.profile_compute_time_every_n_steps is not None and step % \
                self.profile_compute_time_every_n_steps == 0:
            self.run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

    def train_step_end(self, session, global_step, epoch,
            step, loss_labels, loss_recon, loss_total, feed_dict):
        self.log_metrics(epoch=epoch, loss_labels=loss_labels,
            loss_recon=loss_recon, loss_total=loss_total,
            step=step, is_test=False)
        self.log_run_metadata(global_step, step)
        self.log_summaries(session, global_step, step, feed_dict)
        self.tb_writer.flush()

    def validation_step_begin(self, step, number_of_validation_steps):
        self.log_msg("Validating (step {}/{})...".
            format(step + 1, number_of_validation_steps + 1), True)

    def end_epoch(self, session, epoch, global_step, test_loss_labels,
            test_loss_recon, test_loss_total, accuracy):
        self.log_metrics(epoch=epoch, loss_labels=test_loss_labels,
            loss_recon=test_loss_recon, loss_total=test_loss_total,
            accuracy=accuracy, is_test=True)
        self.tb_writer.flush()
        self.save_model_latest(session, epoch, global_step)
        if accuracy >= self.best_accuracy:
            self.best_accuracy = accuracy
            self.save_model_best(session, epoch, global_step)

    def log_metrics(self, epoch, loss_labels, loss_recon,
            loss_total, step=0, is_test=False, accuracy=None):
        prefix = "Test" if is_test else "Train"
        summary = tf.Summary()
        if is_test:
            s_accuracy = summary.value.add()
            s_accuracy.tag = "{}/Accuracy".format(prefix)
            s_accuracy.simple_value = accuracy
        s_loss_labels = summary.value.add()
        s_loss_labels.tag = "{}/Loss(Labels)".format(prefix)
        s_loss_labels.simple_value = loss_labels
        s_loss_recon = summary.value.add()
        s_loss_recon.tag = "{}/Loss(Reconstruction)".format(prefix)
        s_loss_recon.simple_value = loss_recon
        s_loss_total = summary.value.add()
        s_loss_total.tag = "{}/Loss(Total)".format(prefix)
        s_loss_total.simple_value = loss_total

        steps_per_epoch = int(np.ceil(
            self.MNIST_TRAINING_IMAGE_COUNT / BATCH_SIZE))
        step_number = (epoch - 1) * steps_per_epoch + step

        row_dict = dict({
            "Date": datetime.now().strftime("%Y%m%d"),
            "Time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "Epoch": epoch,
            "Step": step_number,
            "Accuracy": accuracy,
            "Loss(Labels)": loss_labels,
            "Loss(Reconstruction)": loss_recon,
            "Loss(Total)": loss_total})

        if is_test:
            self.tb_writer.add_summary(summary, epoch)
            self.tb_writer.flush()
            self.log_msg("[TEST] - Epoch {}, Accuracy {}".format(
                epoch, accuracy))
            del row_dict["Step"]
            self.test_csv_writer.writerow(row_dict)
            self.test_csv.flush()
        else:
            self.tb_writer.add_summary(summary, step_number)
            self.tb_writer.flush()
            self.log_msg("[TRAIN] - Epoch {}, Step {}".format(epoch, step))
            del row_dict["Accuracy"]
            self.train_csv_writer.writerow(row_dict)
            self.train_csv.flush()

        self.log_msg("loss_labels: {}".format(loss_labels), False)
        self.log_msg("loss_reconstruction: {}".format(loss_recon), False)
        self.log_msg("loss: {}".format(loss_total), False)

    def log_run_metadata(self, global_step, step):
        if self.run_metadata is not None and step % \
                self.profile_compute_time_every_n_steps == 0:
            self.tb_writer.add_run_metadata(self.run_metadata,
                "step{}".format(global_step))

    def log_summaries(self, session, global_step, step, feed_dict):
        if self.save_summary_info_every_n_steps is not None and step % \
                self.save_summary_info_every_n_steps == 0:
            summary_op = tf.summary.merge_all()
            summary_str = session.run(summary_op, feed_dict=feed_dict)
            self.tb_writer.add_summary(summary_str, global_step)

    def load_mnist_train(self):
        fd = open(os.path.join(DATA_BASE_DIR, "train-images-idx3-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape([self.MNIST_TRAINING_IMAGE_COUNT,
            28, 28, 1]).astype(np.float32) / 255.
        fd.close()
        fd = open(os.path.join(DATA_BASE_DIR, "train-labels-idx1-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape(
            [self.MNIST_TRAINING_IMAGE_COUNT]).astype(np.int32)
        trainY = np.eye(10)[trainY]
        fd.close()
        return trainX, trainY

    def load_mnist_test(self):
        fd = open(os.path.join(DATA_BASE_DIR, "t10k-images-idx3-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        testX = loaded[16:].reshape([self.MNIST_TESTING_IMAGE_COUNT,
            28, 28, 1]).astype(np.float) / 255.
        fd.close()
        fd = open(os.path.join(DATA_BASE_DIR, "t10k-labels-idx1-ubyte"))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        testY = loaded[8:].reshape(
            [self.MNIST_TESTING_IMAGE_COUNT]).astype(np.int32)
        testY = np.eye(10)[testY]
        fd.close()
        return testX, testY

    def num_training_batches_per_epoch(self):
        return int(np.ceil(self.MNIST_TRAINING_IMAGE_COUNT / BATCH_SIZE))

    def num_validation_batches(self):
        return int(np.ceil(self.MNIST_TESTING_IMAGE_COUNT / BATCH_SIZE))

    @staticmethod
    def open_csv(csv_file_name):
        if os.path.exists(csv_file_name):
            return open(csv_file_name, "a", newline="")
        return open(csv_file_name, "w", newline="")

    @staticmethod
    def log_msg(msg, put_time=True):
        t_str = "                     "
        if put_time:
            t_str = datetime.now().strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        print("{} - {}".format(t_str, msg))
