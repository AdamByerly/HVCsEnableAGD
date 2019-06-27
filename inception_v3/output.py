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

import os
import csv
import tensorflow as tf
from datetime import datetime


class Output:
    CSV_FIELDS = ["Date", "Time", "Epoch",
                  "Top-1 Accuracy", "Top-5 Accuracy", "Loss", "LR"]

    def __init__(self, log_dir, run_name,
                 profile_compute_time_every_n_steps=None,
                 save_summary_info_every_n_steps=None):
        self.run_name         = run_name
        self.pctens           = profile_compute_time_every_n_steps
        self.ssiens           = save_summary_info_every_n_steps
        self.log_dir          = log_dir
        model_file_base       = os.path.join(log_dir, run_name)
        self.model_file_base  = os.path.join(model_file_base, "weights")
        test_csv_filename     = "{}_test_log.csv".format(run_name)
        test_csv_filename     = os.path.join(log_dir, test_csv_filename)
        nbl_test_csv_filename = "{}_nbl_test_log.csv".format(run_name)
        nbl_test_csv_filename = os.path.join(log_dir, nbl_test_csv_filename)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.test_csv               = self.open_csv(test_csv_filename)
        self.nbl_test_csv           = self.open_csv(nbl_test_csv_filename)
        self.test_csv_writer        = csv.DictWriter(self.test_csv,
                                                     fieldnames=self.CSV_FIELDS)
        self.nbl_test_csv_writer    = csv.DictWriter(self.nbl_test_csv,
                                                     fieldnames=self.CSV_FIELDS)
        self.test_csv_writer        .writeheader()
        self.nbl_test_csv_writer    .writeheader()
        self.test_csv               .flush()
        self.nbl_test_csv           .flush()
        self.tb_writer              = None
        self.tb_writer_nbl          = None
        self.tf_saver_best_top1     = None
        self.tf_saver_best_top5     = None
        self.tf_saver_best_nbl_top1 = None
        self.tf_saver_best_nbl_top5 = None
        self.tf_saver_latest        = None
        self.run_options            = None
        self.run_metadata           = None
        self.best_top1_accuracy     = 0
        self.best_top5_accuracy     = 0
        self.best_nbl_top1_accuracy = 0
        self.best_nbl_top5_accuracy = 0

    def train_step_begin(self, step):
        if self.pctens is not None and step % self.pctens == 0:
            self.run_options = tf.RunOptions(
                report_tensor_allocations_upon_oom=True,
                trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

    def train_step_end(self, session, epoch, global_step,
                       step, loss, lr, number_of_training_steps, feed_dict):
        self.log_metrics(epoch=epoch, loss=loss, lr=lr, step=step,
                         steps_per_epoch=number_of_training_steps,
                         is_test=False, validate_nbl=False)
        self.log_run_metadata(global_step, step)
        self.log_summaries(session, global_step, step, feed_dict)
        self.tb_writer.flush()

    def train_end(self, session, epoch, global_step):
        self.save_model_latest(session, epoch, global_step)

    def validation_step_begin(self, step, number_of_validation_steps):
        self.log_msg("Validating (step {}/{})...".
                     format(step + 1, number_of_validation_steps), True)

    def validation_end(self, session, epoch, global_step, validate_nbl,
                       test_loss, lr, top1_accuracy, top5_accuracy):
        self.log_metrics(epoch=epoch, loss=test_loss, lr=lr,
                         top1_accuracy=top1_accuracy,
                         top5_accuracy=top5_accuracy,
                         is_test=True, validate_nbl=validate_nbl)
        self.tb_writer.flush()
        self.tb_writer_nbl.flush()
        if validate_nbl:
            if top1_accuracy >= self.best_nbl_top1_accuracy:
                self.best_nbl_top1_accuracy = top1_accuracy
                self.save_model_nbl_best_top1(session, epoch, global_step)
            if top5_accuracy >= self.best_nbl_top5_accuracy:
                self.best_nbl_top5_accuracy = top5_accuracy
                self.save_model_nbl_best_top5(session, epoch, global_step)
        else:
            if top1_accuracy >= self.best_top1_accuracy:
                self.best_top1_accuracy = top1_accuracy
                self.save_model_best_top1(session, epoch, global_step)
            if top5_accuracy >= self.best_top5_accuracy:
                self.best_top5_accuracy = top5_accuracy
                self.save_model_best_top5(session, epoch, global_step)

    def log_metrics(self, epoch, loss, lr, step=0,
                    steps_per_epoch=0, is_test=False, validate_nbl=False,
                    top1_accuracy=None, top5_accuracy=None):
        prefix = "Test" if is_test else "Train"
        summary = tf.Summary()
        s_loss = summary.value.add()
        s_loss.tag = "{}/Loss".format(prefix)
        s_loss.simple_value = loss
        if is_test:
            s_accuracy1 = summary.value.add()
            s_accuracy1.tag = "{}/Top-1 Accuracy".format(prefix)
            s_accuracy1.simple_value = top1_accuracy
            s_accuracy5 = summary.value.add()
            s_accuracy5.tag = "{}/Top-5 Accuracy".format(prefix)
            s_accuracy5.simple_value = top5_accuracy

        step_number = (epoch - 1) * steps_per_epoch + step

        row_dict = dict({
            "Date": datetime.now().strftime("%Y%m%d"),
            "Time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "Epoch": epoch,
            "Top-1 Accuracy": top1_accuracy,
            "Top-5 Accuracy": top5_accuracy,
            "Loss": loss,
            "LR": lr})

        if is_test:
            if validate_nbl:
                self.tb_writer_nbl.add_summary(summary, epoch)
                self.log_msg("[TEST-NBL] - Epoch {}".format(epoch))
                self.nbl_test_csv_writer.writerow(row_dict)
                self.nbl_test_csv.flush()
            else:
                self.tb_writer.add_summary(summary, epoch)
                self.log_msg("[TEST] - Epoch {}".format(epoch))
                self.test_csv_writer.writerow(row_dict)
                self.test_csv.flush()
        else:
            self.tb_writer.add_summary(summary, step_number)
            self.log_msg("[TRAIN] - Epoch {}, Step {}".format(epoch, step+1))

        self.log_msg("loss: {}".format(loss), False)
        if top1_accuracy is not None:
            self.log_msg("top1: {}".format(top1_accuracy), False)
        if top5_accuracy is not None:
            self.log_msg("top5: {}".format(top5_accuracy), False)

    def log_run_metadata(self, global_step, step):
        if self.run_metadata is not None and step % self.pctens == 0:
            self.tb_writer.add_run_metadata(self.run_metadata,
                                            "step{}".format(global_step))

    def log_summaries(self, session, global_step, step, feed_dict):
        if self.ssiens is not None and step % self.ssiens == 0:
            summary_op = tf.summary.merge_all()
            summary_str = session.run(summary_op, feed_dict=feed_dict)
            self.tb_writer.add_summary(summary_str, global_step)

    def save_model_best_top1(self, session, epoch, g_step):
        self.tf_saver_best_top1.save(session, "{}-{}-best_top1"
                                     .format(self.model_file_base, epoch),
                                     global_step=g_step)

    def save_model_best_top5(self, session, epoch, g_step):
        self.tf_saver_best_top5.save(session, "{}-{}-best_top5"
                                     .format(self.model_file_base, epoch),
                                     global_step=g_step)

    def save_model_nbl_best_top1(self, session, epoch, g_step):
        self.tf_saver_best_nbl_top1.save(session, "{}-{}-nbl_best_top1"
                                         .format(self.model_file_base, epoch),
                                         global_step=g_step)

    def save_model_nbl_best_top5(self, session, epoch, g_step):
        self.tf_saver_best_nbl_top5.save(session, "{}-{}-nbl_best_top5"
                                         .format(self.model_file_base, epoch),
                                         global_step=g_step)

    def save_model_latest(self, session, epoch, g_step):
        self.tf_saver_latest.save(session, "{}-{}-latest"
                                  .format(self.model_file_base, epoch),
                                  global_step=g_step)

    def set_session_graph(self, session_graph):
        self.tb_writer = tf.summary.FileWriter(
            os.path.join(self.log_dir, self.run_name), session_graph)
        self.tb_writer_nbl = tf.summary.FileWriter(
            os.path.join(self.log_dir, self.run_name+"-NBL"), session_graph)
        self.tf_saver_best_top1     = tf.train.Saver(max_to_keep=5)
        self.tf_saver_best_top5     = tf.train.Saver(max_to_keep=5)
        self.tf_saver_best_nbl_top1 = tf.train.Saver(max_to_keep=5)
        self.tf_saver_best_nbl_top5 = tf.train.Saver(max_to_keep=5)
        self.tf_saver_latest        = tf.train.Saver(max_to_keep=5)

    def close_files(self):
        self.test_csv.close()
        self.nbl_test_csv.close()
        self.tb_writer.close()
        self.tb_writer_nbl.close()

    def get_run_options(self):
        return self.run_options

    def get_run_metadata(self):
        return self.run_metadata

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
