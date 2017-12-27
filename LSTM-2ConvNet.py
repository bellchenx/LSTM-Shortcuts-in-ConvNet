import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def rnn_layer(x, timesteps, num_hidden, weights):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell_a = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_cell_b = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_cell_a, lstm_cell_b, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights)

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# ConvLayer 1 with max-pooling
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# ConvLayer 2 with max-pooling
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# LSTM Branch
h_pool1_reshape = tf.transpose(h_pool1, [0, 3, 1, 2])
h_pool1_reshape = tf.reshape(h_pool1_reshape, [-1, 32, 144])
W_lstm = weight_variable([256, 1536])
h_lstm = rnn_layer(h_pool1_reshape, 32, 128, W_lstm)

# Dense Layer 1
W_dense1 = weight_variable([4 * 4 * 64, 1536])
b_dense1 = bias_variable([1536])
h_pool2 = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
h_dense1 = tf.nn.relu(tf.matmul(h_pool2, W_dense1) + h_lstm + b_dense1)

# Dense Layer 2
W_dense2 = weight_variable([1536, 128])
b_dense2 = bias_variable([128])
h_dense2 = tf.nn.relu(tf.matmul(h_dense1, W_dense2) + b_dense2)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_dense2_drop = tf.nn.dropout(h_dense2, keep_prob)

# Dense Layer 3 with Softmax Output
W_dense3 = weight_variable([128, 10])
b_dense3 = bias_variable([10])
y_conv = tf.matmul(h_dense2_drop, W_dense3) + b_dense3

# Training Parameters
training_rate = tf.placeholder(tf.float32)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(training_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    # Training
    sess.run(tf.global_variables_initializer())
    data_location = './MNIST-LSTM-2ConvNet-DATA/MNIST_LSTM_ConvNet'
    saver.restore(sess, data_location)
    last_time = time.time()
    rate = 0.0001
    for i in range(50000):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={
                 x: batch[0], y_: batch[1], keep_prob: 0.5, training_rate: rate})
        if i % 10 == 0:
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={
                                 x: batch[0], y_: batch[1], keep_prob: 1.0, training_rate: rate})
            print('Step: %d, Accuracy: %.2f, Loss: %.5f, Speed: %.1f sec/10 steps' %
                  (i, acc, loss, time.time() - last_time))
            last_time = time.time()                
        if i % 250 == 0:
            current_accuracy = accuracy.eval(
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, training_rate: rate})
            print('- Current Test Accuracy %.4f' % current_accuracy)
            saver.save(sess, data_location)
            print('- Model Saved in Step %d' % i) 
            if current_accuracy > 0.98:
                rate = 0.00003
            if current_accuracy > 0.99:
                rate = 0.00001
            if current_accuracy > 0.993:
                rate = 0.000003
            if current_accuracy > 0.995:
                print('- Accuracy Reached 99.5% in Step %d' % i)
                break 
            last_time = time.time()