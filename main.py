import tensorflow as tf
import os
import numpy as np
from scipy.misc import imresize
import pylab as pl
from PIL import Image

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

images = [x[0] for x in a]
labels = [x[1] for x in a]



LOGDIR = os.path.dirname(__file__)
SAVEPATH = r'model_saves/'
SUMMARYPATH = r'model_summaries/'
SUMDIR = os.path.join(LOGDIR, SUMMARYPATH)
SAVEDIR = os.path.join(LOGDIR, SAVEPATH)

if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

if not os.path.exists(SUMDIR):
    os.mkdir(SUMDIR)


def weight_variable(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

def bias_variable(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


def save_model(save_step, sess, saver):
    save_path = (SAVEDIR)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, (save_path + '/model.ckpt'), global_step=save_step)
    print('Model saved to %s' % save_path)

def neural_network(input, input_size, num_layers, num_neurons, num_classes, activation_function):
    input_channel = input_size
    output_channel = num_neurons
    prev_layer = input

    if num_layers > 1:
        for _ in range(num_layers):
            if activation_function == 'relu':
                out_layer = tf.nn.relu(tf.add(tf.matmul(
                    prev_layer, weight_variable([input_channel, output_channel])),
                    bias_variable([output_channel])))
            elif activation_function == 'sigmoid':
                out_layer = tf.nn.sigmoid(tf.add(tf.matmul(
                    prev_layer, weight_variable([input_channel, output_channel])),
                    bias_variable([output_channel])))
            elif activation_function == 'tanh':
                out_layer = tf.nn.tanh(tf.add(tf.matmul(
                    prev_layer, weight_variable([input_channel, output_channel])),
                    bias_variable([output_channel])))
            else:
                out_layer = tf.add(tf.matmul(
                    prev_layer, weight_variable([input_channel, output_channel])),
                    bias_variable([output_channel]))


            prev_layer = out_layer
            input_channel = output_channel

    if activation_function == 'relu':
        final_layer = tf.nn.relu(tf.add(tf.matmul(
            prev_layer, weight_variable([input_channel, num_classes])),
            bias_variable([num_classes])))
    elif activation_function == 'sigmoid':
        final_layer = tf.nn.sigmoid(tf.add(tf.matmul(
            prev_layer, weight_variable([input_channel, num_classes])),
            bias_variable([num_classes])))
    elif activation_function == 'tanh':
        final_layer = tf.nn.tanh(tf.add(tf.matmul(
            prev_layer, weight_variable([input_channel, num_classes])),
            bias_variable([num_classes])))
    else:
        final_layer = tf.add(tf.matmul(
            prev_layer, weight_variable([input_channel, num_classes])),
            bias_variable([num_classes]))

    return final_layer

def save_model(hparam, save_step, sess, saver):
    save_path = (SAVEDIR + hparam)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, (save_path + '/model.ckpt'), global_step=save_step)
    print('Model saved to %s' % save_path)

def mnist_model(learning_rate,
                batch_size,
                num_epochs,
                display_step,
                num_neurons,
                num_layers,
                input_size,
                num_classes,
                hparam,
                save_step,
                activation_function):

    # Reset the graph
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Output of network
    y = neural_network(x, input_size, num_layers, num_neurons, num_classes, activation_function)

    # Cost function
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('Train'):
        # Gradient descent optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # Back propagation step
        train_step = optimizer.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        # Get prediction and calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge summaries
    summary = tf.summary.merge_all()

    # Setup saver to save variables and graph
    saver = tf.train.Saver()

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMDIR + hparam)
    writer.add_graph(sess.graph)

    # Train
    for step in range(num_epochs):
        batch = mnist.train.next_batch(batch_size)

        # Display accuracy
        if step % display_step == 0:
            [train_accuracy, sum] = sess.run([accuracy, summary], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print('training accuracy: %s, step: %d'%(train_accuracy, step))
            writer.add_summary(sum, step)
        if step % save_step == 0:
            save_model(hparam, step, sess, saver)

        # Training step
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

def make_hparam_string(batch_size, learning_rate, num_neurons, num_layers, activation_function):
    return "batch=%d_lr_%.0E_neurons=%s_layers=%s_afunc=%s" % (batch_size, learning_rate, num_neurons, num_layers, activation_function)


def main():
    # Parameters
    num_epochs = 20000
    display_step = 50
    save_step = 2500

    # Network Parameters
    input_size = 40000  # data input (img shape: 200*200)
    num_classes = 2  # classes (NORMAL, PNEUMONIA)

    # Parameter search
    batch_size = 5
    neurons = 50
    layers = 2
    learning_rate = [0.1, 0.3, 0.5]
    activation_function = 'relu'

    hparam = make_hparam_string(batch_size, learning_rate, neurons, layers, activation_function)
    print('Starting run for %s' % hparam)

    mnist_model(learning_rate,
                batch_size,
                num_epochs,
                display_step,
                neurons,
                layers,
                input_size,
                num_classes,
                hparam,
                save_step,
                activation_function)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % SUMDIR)



if __name__ == '__main__':
    main()
