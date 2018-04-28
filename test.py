import tensorflow as tf
import os


LOGDIR = os.path.dirname(__file__)
SAVEPATH = r'model_saves/'
SUMMARYPATH = r'model_summaries/'
SUMDIR = os.path.join(LOGDIR, SUMMARYPATH)
SAVEDIR = os.path.join(LOGDIR, SAVEPATH)

if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

if not os.path.exists(SUMDIR):
    os.mkdir(SUMDIR)

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

def weight_variable(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

def bias_variable(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)

def neural_network(input, num_input, num_layers, num_neurons, num_classes, activation_function):
    input_channel = num_input
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
                num_input,
                num_classes,
                hparam,
                save_step,
                activation_function):

    # Reset the graph
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, num_input])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

    # Output of network
    y = neural_network(x, num_input, num_layers, num_neurons, num_classes, activation_function)

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
    display_step = 100
    save_step = 2500

    # Network Parameters
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)

    # Parameter search
    batch_sizes = [50]
    num_of_neurons = [10,20,50]
    num_of_layers = [1,2,5]
    learning_rates = [0.1,0.3,0.5]
    activation_functions = ['relu','sigmoid','tanh']

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for neurons in num_of_neurons:
                for layers in num_of_layers:
                    for activation_function in activation_functions:
                        hparam = make_hparam_string(batch_size, learning_rate, neurons, layers, activation_function)
                        print('Starting run for %s'%hparam)

                        mnist_model(learning_rate,
                                    batch_size,
                                    num_epochs,
                                    display_step,
                                    neurons,
                                    layers,
                                    num_input,
                                    num_classes,
                                    hparam,
                                    save_step,
                                    activation_function)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % SUMDIR)

if __name__ == '__main__':
    main()