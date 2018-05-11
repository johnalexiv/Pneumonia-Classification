import tensorflow as tf
import os
import numpy as np
from scipy.misc import imresize
import random
from PIL import Image
from time import gmtime, strftime
import sys

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class DataSet():
    def __init__(self):
        self.data_dir = os.getcwd() + '/chest_xray'
        self.train_dir = self.data_dir + '/train'
        self.val_dir = self.data_dir + '/val'
        self.test_dir = self.data_dir + '/test'
        self.counters = {'train_normal_counter': 0,
                         'train_pneumonia_counter': 0,
                         'val_normal_counter': 0,
                         'val_pneumonia_counter': 0,
                         'test_normal_counter': 0,
                         'test_pneumonia_counter': 0}
        self.shuffle = True
        self.skipped = 0

        self.train_normal_files = os.listdir(self.train_dir + '/NORMAL')
        self.train_pneumonia_files = os.listdir(self.train_dir + '/PNEUMONIA')

        self.test_normal_files = os.listdir(self.test_dir + '/NORMAL')
        self.test_pneumonia_files = os.listdir(self.test_dir + '/PNEUMONIA')

        self.val_normal_files = os.listdir(self.val_dir + '/NORMAL')
        self.val_pneumonia_files = os.listdir(self.val_dir + '/PNEUMONIA')

        self.prepare_data()

    def add_data(self, dir, files, images):

        progress = 0
        print_progress(progress, len(files), prefix=dir, suffix='Complete', bar_length=50)

        for file in files:
            if not '.jpeg' in file:
                continue
            image = self.crop_resize_image(os.path.join(dir, file))

            if image is None:
                self.skipped += 1
                continue

            if 'NORMAL' in dir:
                images.append((image, 0))
            else:
                images.append((image, 1))

            progress += 1
            print_progress(progress, len(files), prefix=dir, suffix='Complete', bar_length=50)



    def prepare_data(self):
        self.train_normal_images = []
        self.train_pneumonia_images = []

        self.val_normal_images = []
        self.val_pneumonia_images = []

        self.test_normal_images = []
        self.test_pneumonia_images = []

        train_balance = min(len(self.train_normal_files), len(self.train_pneumonia_files))
        val_balance = min(len(self.val_normal_files), len(self.val_pneumonia_files))
        test_balance = min(len(self.test_normal_files), len(self.test_pneumonia_files))

        # Load training data
        self.add_data(self.train_dir + '/NORMAL', self.train_normal_files[:train_balance], self.train_normal_images)
        self.add_data(self.train_dir + '/PNEUMONIA', self.train_pneumonia_files[:train_balance], self.train_pneumonia_images)
        self.add_data(self.val_dir + '/NORMAL', self.val_normal_files[:val_balance], self.val_normal_images)
        self.add_data(self.val_dir + '/PNEUMONIA', self.val_pneumonia_files[:val_balance], self.val_pneumonia_images)
        self.add_data(self.test_dir + '/NORMAL', self.test_normal_files[:test_balance], self.test_normal_images)
        self.add_data(self.test_dir + '/PNEUMONIA', self.test_pneumonia_files[:test_balance], self.test_pneumonia_images)

        print('Skipped: %d' % self.skipped)

    def crop_resize_image(self, file_path):
        try:
            img = Image.open(file_path)
            # print('File: %s, Shape: %s'% (file_path, img.size))

            diff = abs(img.size[0] - img.size[1])
            odd = False
            if diff % 2 == 1:
                odd = True
            padding = int(diff / 2)
            if img.size[0] > img.size[1]:
                if odd:
                    new_img = np.pad(img, [(padding, padding + 1), (0, 0)], mode='constant')
                else:
                    new_img = np.pad(img, [(padding, padding), (0, 0)], mode='constant')
            else:
                if odd:
                    new_img = np.pad(img, [(0, 0), (padding, padding + 1)], mode='constant')
                else:
                    new_img = np.pad(img, [(0, 0), (padding, padding)], mode='constant')

            new_img = imresize(new_img, (256, 256), interp='bilinear')
        except:
            # print('Failed to load: File: %s' % file_path)
            return None

        return np.asarray(new_img)

    def add_images(self, batch_size, counter, data):
        reset_counter = False
        images = []

        if counter == 'train_n':
            key = 'train_normal_counter'
        elif counter == 'train_p':
            key = 'train_pneumonia_counter'
        elif counter == 'test_n':
            key = 'test_normal_counter'
        elif counter == 'test_p':
            key = 'test_pneumonia_counter'
        elif counter =='val_n':
            key = 'val_normal_counter'
        elif counter == 'val_p':
            key = 'val_pneumonia_counter'

        batch_limit = self.counters[key] + batch_size

        if batch_size < 0:
            self.counters[key] = 0
            batch_limit = len(data)

        if self.shuffle:
            random.shuffle(data)
            self.shuffle = False

        if batch_limit > len(data):
            batch_limit = len(data)
            reset_counter = True
            self.shuffle = True

        while self.counters[key] < batch_limit:
            label = np.zeros(2).astype(np.int32)
            label[data[self.counters[key]][1]] = 1
            images.append((data[self.counters[key]][0], label))
            self.counters[key] += 1

        if reset_counter:
            self.counters[key] = 0

        return images

    def next_batch(self, batch_size, type):
        normal = []
        pneumonia = []

        if type == 'train':
            normal = self.add_images(batch_size=(batch_size / 2),
                                     counter='train_n',
                                     data=self.train_normal_images)
            pneumonia = self.add_images(batch_size=(batch_size / 2),
                                        counter='train_p',
                                        data=self.train_pneumonia_images)
        elif type == 'test':
            normal = self.add_images(batch_size=(batch_size / 2),
                                     counter='test_n',
                                     data=self.test_normal_images)
            pneumonia = self.add_images(batch_size=(batch_size / 2),
                                        counter='test_p',
                                        data=self.test_pneumonia_images)
        elif type == 'val':
            normal = self.add_images(batch_size=(batch_size / 2),
                                     counter='test_n',
                                     data=self.val_normal_images)
            pneumonia = self.add_images(batch_size=(batch_size / 2),
                                        counter='test_p',
                                        data=self.val_pneumonia_images)

        normal.extend(pneumonia)
        random.shuffle(normal)

        images = [x[0] for x in normal]
        labels = [x[1] for x in normal]

        return (np.reshape(np.array(images), (len(images), -1)), np.array(labels))


LOGDIR = os.path.dirname(__file__)
SAVEPATH = r'cnn/model_saves/'
SUMMARYPATH = r'cnn/model_summaries/'
SUMDIR = os.path.join(LOGDIR, SUMMARYPATH)
SAVEDIR = os.path.join(LOGDIR, SAVEPATH)

if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

if not os.path.exists(SUMDIR):
    os.mkdir(SUMDIR)


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def save_model(save_step, sess, saver):
    save_path = (SAVEDIR)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, (save_path + '/model.ckpt'), global_step=save_step)
    sys.stdout.write('Model saved to %s' % save_path)
    sys.stdout.flush()

def cnn(image):
    with tf.name_scope('image_reshape'):
        # reshape data
        input = tf.reshape(image, [-1, 256, 256, 1])

    with tf.name_scope('Conv1'):
        # First layer (256 x 256 x 1)
        W_conv1 = weight_variable([3, 3, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(input, W_conv1) + b_conv1)

    with tf.name_scope('Pool1'):
        pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv2'):
        # Second Layer (128 x 128 x 16)
        W_conv2 = weight_variable([3, 3, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)

    with tf.name_scope('Pool2'):
        pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('Conv3'):
        # Third Layer (64 x 64 x 32)
        W_conv3 = weight_variable([3, 3, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(pool2, W_conv3) + b_conv3)

    with tf.name_scope('Pool3'):
        pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('Conv4'):
        # Fourth Layer (32 x 32 x 64)
        W_conv4 = weight_variable([3, 3, 64, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(pool3, W_conv4) + b_conv4)

    with tf.name_scope('Pool4'):
        pool4 = max_pool_2x2(h_conv4)

    with tf.name_scope('Conv5'):
        # Fifth Layer (16 x 16 x 128)
        W_conv5 = weight_variable([3, 3, 128, 256])
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(pool4, W_conv5) + b_conv5)

    with tf.name_scope('Pool5'):
        pool5 = max_pool_2x2(h_conv5)

    with tf.name_scope('Fc1'):
        # fully connected layer 1 (8 x 8 x 256)
        W_fc1 = weight_variable([8 * 8 * 256, 1024])
        b_fc1 = bias_variable([1024])

    with tf.name_scope('flatten'):
        h_pool5_flat = tf.reshape(pool5, [-1, 8 * 8 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        # Drop out
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('output'):
        # output layer
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])

        output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return output, keep_prob

def model(learning_rate, num_epochs, image_size, num_classes, hparam, save_step):
    # Input Layer
    x = tf.placeholder(tf.float32, shape=[None, image_size])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    y, keep_prob = cnn(x)

    # Cost function
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('Train'):
        # Gradient descent optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Back propagation step
        train_step = optimizer.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        # Get prediction and calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_acc = tf.summary.scalar('train_accuracy', accuracy)
        val_acc = tf.summary.scalar('test_accuracy', accuracy)

    # Setup saver to save variables and graph
    saver = tf.train.Saver()

    # Initialize all variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMDIR + hparam)
    writer.add_graph(sess.graph)

    dataset = DataSet()

    for step in range(num_epochs):
        batch_xs, batch_ys = dataset.next_batch(20, 'train')
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        if step % 5 == 0:
            [train_accuracy, sum] = sess.run([accuracy, train_acc], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            sys.stdout.write('training accuracy: %s, step: %d \n' % (train_accuracy, step))
            sys.stdout.flush()
            writer.add_summary(sum, step)

        # Display accuracy
        if step % 100 == 0:
            batch_xs, batch_ys = dataset.next_batch(-1, 'val')
            [val_accuracy, sum] = sess.run([accuracy, val_acc], feed_dict={x: batch_xs, y_: batch_ys})
            sys.stdout.write('validation accuracy: %s, step: %d \n' % (val_accuracy, step))
            sys.stdout.flush()
            writer.add_summary(sum, step)
        if step % save_step == 0:
            save_model(step, sess, saver)

    batch_xs, batch_ys = dataset.next_batch(-1, 'test')
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    sys.stdout.write('testing accuracy: %s \n' % test_accuracy)


def make_hparam_string():
    return strftime("%Y-%m-%d %H.%M.%S", gmtime())

def main():
    # Parameters
    num_epochs = 5000
    save_step = 1000

    # Network Parameters
    image_size = 256 * 256  # data input (img shape: 256*256)
    num_classes = 2  # classes (NORMAL, PNEUMONIA)

    learning_rate = 0.0001

    hparam = make_hparam_string()
    print('Starting run for %s' % hparam)


    model(learning_rate,
            num_epochs,
            image_size,
            num_classes,
            hparam,
            save_step)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % SUMDIR)


if __name__ == '__main__':
    main()