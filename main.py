import tensorflow as tf
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_dir = os.getcwd() + '/chest_xray'

train_normal_files = os.listdir(data_dir + '/train/NORMAL')
train_pneumonia_files = os.listdir(data_dir + '/train/PNEUMONIA')

test_normal_files = os.listdir(data_dir + '/test/NORMAL')
test_pneumonia_files = os.listdir(data_dir + '/test/PNEUMONIA')

val_normal_files = os.listdir(data_dir + '/val/NORMAL')
val_pneumonia_files = os.listdir(data_dir + '/val/PNEUMONIA')

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

def neural_network(input, num_input, num_layers, num_neurons, num_classes):
    pass

def cnn(input, num_input, num_layers, num_neurons, num_classes):
    pass

def load_dataset():
    pass

def model(learning_rate,
            batch_size,
            num_epochs,
            display_step,
            num_input,
            num_classes):

    # Reset the graph
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, num_input])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Output of network
    y = neural_network(x)

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
    writer = tf.summary.FileWriter(SUMDIR)
    writer.add_graph(sess.graph)

    # Train
    for step in range(num_epochs):
        # get data from files
        xray_images = load_dataset()

        # Display accuracy
        if step % display_step == 0:
            [train_accuracy, sum] = sess.run([accuracy, summary], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print('training accuracy: %s, step: %d'%(train_accuracy, step))
            writer.add_summary(sum, step)
        if step % 100 == 0:
            save_model(step, sess, saver)

        # Training step
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


for i in range(1):
    image_data = mpimg.imread(os.path.join(data_dir, 'train/NORMAL', train_normal_files[0]))

plt.imshow(image_data)
plt.show()