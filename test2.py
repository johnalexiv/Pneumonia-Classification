# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
from scipy.misc import imresize
from PIL import Image


import tensorflow as tf

FLAGS = None


class DataSet():
    def __init__(self):
        self.data_dir = os.getcwd() + '/chest_xray'
        self.train_dir = self.data_dir + '/train'
        self.val_dir = self.data_dir + '/val'
        self.test_dir = self.data_dir + '/test'

        self.train_normal_files = os.listdir(self.train_dir + '/NORMAL')
        self.train_pneumonia_files = os.listdir(self.train_dir + '/PNEUMONIA')

        self.test_normal_files = os.listdir(self.test_dir + '/NORMAL')
        self.test_pneumonia_files = os.listdir(self.test_dir + '/PNEUMONIA')

        self.val_normal_files = os.listdir(self.val_dir + '/NORMAL')
        self.val_pneumonia_files = os.listdir(self.val_dir + '/PNEUMONIA')

        self.prepare_data()

    def prepare_data(self):
        self.train_images = []
        self.val_images = []
        self.test_images = []

        # Load training data
        for file in self.train_normal_files:
            self.train_images.append((os.path.join(self.train_dir + '/NORMAL', file), 0))
        for file in self.train_pneumonia_files:
            self.train_images.append((os.path.join(self.train_dir + '/PNEUMONIA', file), 1))

        # Load validation data
        for file in self.val_normal_files:
            self.val_images.append((os.path.join(self.val_dir + '/NORMAL', file), 0))
        for file in self.val_pneumonia_files:
            self.val_images.append((os.path.join(self.val_dir + '/PNEUMONIA', file), 1))

        # Load test data
        for file in self.test_normal_files:
            self.test_images.append((os.path.join(self.test_dir + '/NORMAL', file), 0))
        for file in self.test_pneumonia_files:
            self.test_images.append((os.path.join(self.test_dir + '/PNEUMONIA', file), 1))

    def crop_resize_image(self, file_path):
        img = Image.open(file_path)

        diff = abs(img.size[0] - img.size[1])
        padding = int(diff / 2)
        if img.size[0] > img.size[1]:
            new_img = np.pad(img, [(padding, padding + 1), (0, 0)], mode='constant')
        else:
            new_img = np.pad(img, [(0, 0), (padding, padding + 1)], mode='constant')

        new_img = imresize(new_img, (200, 200), interp='bilinear')

        return np.asarray(new_img)

    def next_image(self, type):
        pass

    def next_batch(self, batch_size, type):
        batch = []
        for _ in range(batch_size):
            pass


dataset = DataSet()

import random

a = dataset.test_images
random.shuffle(a)

image_files = [x[0] for x in a]
label_values = [x[1] for x in a]

def crop_resize_image(file_path):
    img = Image.open(file_path)

    diff = abs(img.size[0] - img.size[1])
    padding = int(diff / 2)
    if img.size[0] > img.size[1]:
        new_img = np.pad(img, [(padding, padding + 1), (0, 0)], mode='constant')
    else:
        new_img = np.pad(img, [(0, 0), (padding, padding + 1)], mode='constant')

    new_img = imresize(new_img, (200, 200), interp='bilinear')

    return np.asarray(new_img)

images = []
labels = []
for i, file in enumerate(image_files):
    label = np.zeros(2).astype(np.int32)
    label[label_values[i]] = 1
    labels.append(label)
    images.append(crop_resize_image(file))

batch = (np.reshape(np.array(images), (len(images), -1)), np.array(labels))



def main(_):
  # Import data

  # Create the model
  x = tf.placeholder(tf.float32, [None, 40000])
  W = tf.Variable(tf.zeros([40000, 2]))
  b = tf.Variable(tf.zeros([2]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  for i in range(5000):
    batch_xs, batch_ys = batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



  # Test trained model
    if i % 10 == 0:
        print(sess.run(accuracy, feed_dict={x: batch_xs,
                                      y_: batch_ys}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)