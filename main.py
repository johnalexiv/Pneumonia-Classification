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

for i, file in enumerate(image_files[:5]):
    label = np.zeros(2).astype(np.int32)
    label[label_values[i]] = 1
    labels.append(label)
    images.append(crop_resize_image(file))

batch = (np.reshape(np.array(images), (5, -1)), np.array(labels))

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

def neural_network(input, input_size, num_classes):
    x = tf.matmul(input, weight_variable([input_size, 32])) + bias_variable([32])
    x = tf.nn.relu(x)

    x = tf.matmul(x, weight_variable([32, 64])) + bias_variable([64])
    x = tf.nn.relu(x)

    x = tf.matmul(x, weight_variable([64, num_classes])) + bias_variable([num_classes])
    x = tf.nn.relu(x)

    return x

def save_model(hparam, save_step, sess, saver):
    save_path = (SAVEDIR + hparam)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, (save_path + '/model.ckpt'), global_step=save_step)
    print('Model saved to %s' % save_path)

def mnist_model(learning_rate,
                num_epochs,
                input_size,
                num_classes,
                hparam,
                save_step):

    # Reset the graph
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Output of network
    y = neural_network(x, input_size, num_classes)

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
        # tf.summary.scalar('accuracy', accuracy)

    # Merge summaries
    # summary = tf.summary.merge_all()

    # Setup saver to save variables and graph
    # saver = tf.train.Saver()

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMDIR + hparam)
    # writer.add_graph(sess.graph)

    # Train
    for step in range(num_epochs):
        # batch = mnist.train.next_batch(batch_size)

        # Display accuracy
        # if step % display_step == 0:
        #     [train_accuracy, sum] = sess.run([accuracy, summary], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        #     print('training accuracy: %s, step: %d'%(train_accuracy, step))
        #     writer.add_summary(sum, step)
        # if step % save_step == 0:
            # save_model(hparam, step, sess, saver)

        # Training step
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

def make_hparam_string(batch_size, learning_rate, num_neurons, num_layers, activation_function):
    return "batch=%d_lr_%.0E_neurons=%s_layers=%s_afunc=%s" % (batch_size, learning_rate, num_neurons, num_layers, activation_function)


def main():
    # Parameters
    num_epochs = 1
    display_step = 50
    save_step = 2500

    # Network Parameters
    input_size = 40000  # data input (img shape: 200*200)
    num_classes = 2  # classes (NORMAL, PNEUMONIA)

    # Parameter search
    batch_size = 5
    neurons = 50
    layers = 1
    learning_rate = 0.3
    activation_function = 'relu'

    hparam = make_hparam_string(batch_size, learning_rate, neurons, layers, activation_function)
    print('Starting run for %s' % hparam)

    mnist_model(learning_rate,
                num_epochs,
                input_size,
                num_classes,
                hparam,
                save_step)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % SUMDIR)



if __name__ == '__main__':
    main()
