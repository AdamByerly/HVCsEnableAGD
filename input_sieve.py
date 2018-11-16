# Copyright 2018 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# The above copyright notice is applied in accordance with the license
#  of the codebase from which the following was derived.
# That code is Copyright 2016 Google, Inc. and was retrieved from:
#  https://github.com/tensorflow/models/blob/master
#  /research/inception/inception/image_processing.py

import os
import cv2
import math
import numpy as np
import tensorflow as tf

BATCH_SIZE                = 128
IMAGE_WIDTH               = 224
IMAGE_HEIGHT              = 224
PREPROCESS_THREADS        = 24
READERS                   = 16
INPUT_QUEUE_MEMORY_FACTOR = 16
EXAMPLES_PER_SHARD        = 1024
CLASSES                   = 1000
TRAIN_IMAGE_COUNT         = 1281167
VALIDATION_IMAGE_COUNT    = 50000
BASE_DIR                  = "C:\\Users\\Adam\\Downloads\\"
DATA_DIR                  = os.path.join(BASE_DIR,
                                "ILSVRC2017_CLS-LOC\\Data\\CLS-LOC\\processed")


class DataSet(object):
    def __init__(self, subset):
        self.subset = subset

    @staticmethod
    def num_classes():
        return CLASSES

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return TRAIN_IMAGE_COUNT
        if self.subset == 'validation':
            return VALIDATION_IMAGE_COUNT

    def num_batches_per_epoch(self):
        return int(math.ceil(self.num_examples_per_epoch()/BATCH_SIZE))

    def data_files(self):
        tf_record_pattern = os.path.join(DATA_DIR, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        return data_files


def train_inputs(dataset, log_annotated_images=False):
    return batch_inputs(dataset,
        log_annotated_images, True, READERS, "input/batch_train")


def eval_inputs(dataset, log_annotated_images=False):
    return batch_inputs(dataset,
        log_annotated_images, False, 1, "input/batch_eval")


def batch_inputs(dataset, log_annotated_images, train, num_readers, scope_name):
    with tf.device('/cpu:0'), tf.name_scope(scope_name):
        data_files = dataset.data_files()

        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=train,
            capacity=INPUT_QUEUE_MEMORY_FACTOR*2 if train else 1)

        batch_size             = BATCH_SIZE
        num_preprocess_threads = PREPROCESS_THREADS
        examples_per_shard     = EXAMPLES_PER_SHARD
        min_queue_examples     = examples_per_shard*INPUT_QUEUE_MEMORY_FACTOR

        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples+3*batch_size,
                min_after_dequeue=min_queue_examples, dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard+3*batch_size, dtypes=[tf.string])

        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            image_buffer, label_index, bbox, text, synset =\
                parse_example_proto(example_serialized)
            image = image_preprocessing(image_buffer, bbox,
                label_index, text, synset,
                log_annotated_images, train, thread_id)
            images_and_labels.append([image, label_index, text, synset])

        images, labels, texts, synsets = tf.train.batch_join(
            images_and_labels, batch_size=batch_size,
            capacity=2*num_preprocess_threads*batch_size)

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[
            batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        labels = tf.reshape(labels, [batch_size])

        return images, tf.one_hot(labels, CLASSES)


def annotate_images(imgs, labels, texts, synsets):
    image_copy = np.copy(imgs)
    for i in range(image_copy.shape[0]):
        label = labels[i]
        text  = texts[i]
        syn   = synsets[i]
        if isinstance(labels, bytes):
            label = label.decode()
        if isinstance(text, bytes):
            text = text.decode()
        if isinstance(syn, bytes):
            syn = syn.decode()
        cv2.putText(image_copy[i, :, :, :],
            str(label), (0, 18), cv2.FONT_HERSHEY_COMPLEX, .5, (1, 1, 0), 2)
        cv2.putText(image_copy[i, :, :, :],
            str(text), (0, 36), cv2.FONT_HERSHEY_COMPLEX, .5, (1, 1, 0), 2)
        cv2.putText(image_copy[i, :, :, :],
            str(syn), (0, 54), cv2.FONT_HERSHEY_COMPLEX, .5, (1, 1, 0), 2)
    return image_copy


def image_preprocessing(image_buffer, bbox, label, text, synset,
        log_annotated_images, train, thread_id=0):
    image = decode_jpeg(image_buffer)
    if train:
        image = distort_image(image, log_annotated_images,
            IMAGE_HEIGHT, IMAGE_WIDTH, bbox, thread_id)
    else:
        image = eval_image(image, IMAGE_HEIGHT, IMAGE_WIDTH)

    if log_annotated_images and not thread_id:
        labeled_images = tf.py_func(annotate_images,
            [tf.expand_dims(image, 0),
             tf.expand_dims(label, 0),
             tf.expand_dims(text, 0),
             tf.expand_dims(synset, 0)], Tout=image.dtype)
        tf.summary.image('annotated_images', labeled_images)

    # rescale to [-1,1] instead of [0, 1)
    return tf.multiply(tf.subtract(image, 0.5), 2.0)


def eval_image(image, height, width, scope=None):
    with tf.name_scope(values=[image, height, width],
            name=scope, default_name='eval_image'):
        image = tf.image.central_crop(image, central_fraction=0.875)
        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                    align_corners=False)
        return tf.squeeze(image, [0])


def decode_jpeg(image_buffer, scope=None):
    with tf.name_scope(values=[image_buffer],
            name=scope, default_name='decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        return tf.image.convert_image_dtype(image, dtype=tf.float32)


def distort_color(image, thread_id=0, scope=None):
    with tf.name_scope(values=[image],
            name=scope, default_name='distort_color'):
        if thread_id % 2 == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        else:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        return tf.clip_by_value(image, 0.0, 1.0)


def distort_image(image, log_annotated_images,
        height, width, bbox, thread_id=0, scope=None):
    with tf.name_scope(values=[image, height, width, bbox],
            name=scope, default_name='distort_image'):

        if log_annotated_images and not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), bbox)
            tf.summary.image('image_with_bounding_boxes', image_with_box)

        bbox_begin, bbox_size, distort_bbox =\
            tf.image.sample_distorted_bounding_box(
                tf.shape(image), bounding_boxes=bbox, min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0],
                max_attempts=100, use_image_if_no_bounding_boxes=True)

        if log_annotated_images and not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('images_with_distorted_bounding_box',
                image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(
            distorted_image, [height, width], method=resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size
        # loses the third dimension.
        distorted_image.set_shape([height, width, 3])

        if log_annotated_images and not thread_id:
            tf.summary.image('cropped_resized_image',
                tf.expand_dims(distorted_image, 0))

        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = distort_color(distorted_image, thread_id)

        if log_annotated_images and not thread_id:
            tf.summary.image('final_distorted_image',
                tf.expand_dims(distorted_image, 0))

        return distorted_image


def parse_example_proto(example_serialized):
    feature_map = {
        'image/encoded'     : tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label' : tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text'  : tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/synset': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    feature_map.update({k: sparse_float32 for k in [
        'image/object/bbox/xmin', 'image/object/bbox/ymin',
        'image/object/bbox/xmax', 'image/object/bbox/ymax']})
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    return features['image/encoded'], label, bbox,\
           features['image/class/text'], features['image/class/synset']
