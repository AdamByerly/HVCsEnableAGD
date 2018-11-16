import os
import csv
import tensorflow as tf
from datetime import datetime


class Output:
    CSV_FIELDS = ["Date", "Time", "Epoch", "Step",
                  "Top-1 Accuracy", "Top-5 Accuracy", "Loss"]

    def __init__(self, run_name, profile_compute_time_every_n_steps=None,
                 save_summary_info_every_n_steps=None):
        self.run_name = run_name
        self.profile_compute_time_every_n_steps\
            = profile_compute_time_every_n_steps
        self.save_summary_info_every_n_steps\
            = save_summary_info_every_n_steps
        self.model_file_base = os.path.join(
            os.path.join("logs", run_name), "weights")
        self.train_csv = self.open_csv(os.path.join(
            "logs", "{}_training_log.csv".format(run_name)))
        self.test_csv = self.open_csv(os.path.join(
            "logs", "{}_test_log.csv".format(run_name)))
        no_accuracy = self.CSV_FIELDS.copy()
        no_accuracy.remove("Top-1 Accuracy")
        no_accuracy.remove("Top-5 Accuracy")
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
        self.tf_saver_best_top1 = None
        self.tf_saver_best_top5 = None
        self.tf_saver_latest = None
        self.run_options = None
        self.run_metadata = None
        self.best_top1_accuracy = 0
        self.best_top5_accuracy = 0

    def get_run_options(self):
        return self.run_options

    def get_run_metadata(self):
        return self.run_metadata

    def set_session_graph(self, session_graph):
        self.tb_writer = tf.summary.FileWriter(
            os.path.join("logs", self.run_name), session_graph)
        self.tf_saver_best_top1 = tf.train.Saver(max_to_keep=5)
        self.tf_saver_best_top5 = tf.train.Saver(max_to_keep=5)
        self.tf_saver_latest = tf.train.Saver(max_to_keep=5)

    def close_files(self):
        self.train_csv.close()
        self.test_csv.close()
        self.tb_writer.close()

    def save_model_best_top1(self, session, epoch, g_step):
        self.tf_saver_best_top1.save(session, "{}-{}-best_top1"
            .format(self.model_file_base, epoch), global_step=g_step)

    def save_model_best_top5(self, session, epoch, g_step):
        self.tf_saver_best_top5.save(session, "{}-{}-best_top5"
            .format(self.model_file_base, epoch), global_step=g_step)

    def save_model_latest(self, session, epoch, g_step):
        self.tf_saver_latest.save(session, "{}-{}-latest"
            .format(self.model_file_base, epoch), global_step=g_step)

    def train_step_begin(self, step):
        if self.profile_compute_time_every_n_steps is not None and step % \
                self.profile_compute_time_every_n_steps == 0:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

    def train_step_end(self, session, global_step,
            epoch, step, loss, number_of_training_steps, feed_dict):
        self.log_metrics(epoch=epoch, loss=loss, step=step,
            steps_per_epoch=number_of_training_steps, is_test=False)
        self.log_run_metadata(global_step, step)
        self.log_summaries(session, global_step, step, feed_dict)
        self.tb_writer.flush()

    def validation_step_begin(self, step, number_of_validation_steps):
        self.log_msg("Validating (step {}/{})...".
            format(step + 1, number_of_validation_steps + 1), True)

    def end_epoch(self, session, epoch, global_step,
            test_loss, top1_accuracy, top5_accuracy):
        self.log_metrics(epoch=epoch, loss=test_loss,
            top1_accuracy=top1_accuracy, top5_accuracy=top5_accuracy,
            is_test=True)
        self.tb_writer.flush()
        self.save_model_latest(session, epoch, global_step)
        if top1_accuracy >= self.best_top1_accuracy:
            self.best_top1_accuracy = top1_accuracy
            self.save_model_best_top1(session, epoch, global_step)
        if top5_accuracy >= self.best_top5_accuracy:
            self.best_top5_accuracy = top5_accuracy
            self.save_model_best_top5(session, epoch, global_step)

    def log_metrics(self, epoch, loss, step=0, steps_per_epoch=0,
            is_test=False, top1_accuracy=None, top5_accuracy=None):
        prefix = "Test" if is_test else "Train"
        summary = tf.Summary()
        if is_test:
            s_accuracy1 = summary.value.add()
            s_accuracy1.tag = "{}/Top-1 Accuracy".format(prefix)
            s_accuracy1.simple_value = top1_accuracy
            s_accuracy5 = summary.value.add()
            s_accuracy5.tag = "{}/Top-5 Accuracy".format(prefix)
            s_accuracy5.simple_value = top5_accuracy
        s_loss = summary.value.add()
        s_loss.tag = "{}/Loss(Total)".format(prefix)
        s_loss.simple_value = loss

        step_number = (epoch - 1) * steps_per_epoch + step

        row_dict = dict({
            "Date": datetime.now().strftime("%Y%m%d"),
            "Time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "Epoch": epoch,
            "Step": step_number,
            "Top-1 Accuracy": top1_accuracy,
            "Top-5 Accuracy": top5_accuracy,
            "Loss": loss})

        if is_test:
            self.tb_writer.add_summary(summary, epoch)
            self.log_msg("[TEST] - Epoch {}".format(epoch))
            del row_dict["Step"]
            self.test_csv_writer.writerow(row_dict)
            self.test_csv.flush()
        else:
            self.tb_writer.add_summary(summary, step_number)
            self.log_msg("[TRAIN] - Epoch {}, Step {}".format(epoch, step))
            del row_dict["Top-1 Accuracy"]
            del row_dict["Top-5 Accuracy"]
            self.train_csv_writer.writerow(row_dict)
            self.train_csv.flush()

        self.log_msg("loss: {}".format(loss), False)
        if top1_accuracy is not None:
            self.log_msg("top1: {}".format(top1_accuracy), False)
        if top5_accuracy is not None:
            self.log_msg("top5: {}".format(top5_accuracy), False)

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
