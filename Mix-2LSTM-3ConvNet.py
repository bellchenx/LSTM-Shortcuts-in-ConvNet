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

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# ConvLayer 1 with max-pooling
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# ConvLayer 2 with max-pooling
W_conv2 = weight_variable([4, 4, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ConvLayer 3 without pooling
W_conv3 = weight_variable([3, 3, 128, 192])
b_conv3 = bias_variable([192])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# LSTM Branch from ConvLayer 2
with tf.variable_scope('LSTM2'):
    lstm2_num_hidden = 256
    lstm2_timesteps = 128
    h_conv2_reshape = tf.transpose(h_conv2, [0, 3, 1, 2])
    h_conv2_reshape = tf.reshape(h_conv2_reshape, [-1, lstm2_timesteps, 81])
    W_lstm2 = weight_variable([2 * lstm2_num_hidden, 1536])
    h_conv2_reshape = tf.unstack(h_conv2_reshape, lstm2_timesteps, 1)
    lstm2_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm2_num_hidden, forget_bias=1.0)
    lstm2_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm2_num_hidden, forget_bias=1.0)
    lstm2_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm2_cell_fw, lstm2_cell_bw, h_conv2_reshape, dtype=tf.float32)
h_lstm2 = tf.matmul(lstm2_outputs[-1], W_lstm2)

# LSTM Branch from ConvLayer 1
with tf.variable_scope('LSTM1'):
    lstm1_num_hidden = 128
    lstm1_timesteps = 64
    h_conv1_reshape = tf.transpose(h_conv1, [0, 3, 1, 2])
    h_conv1_reshape = tf.reshape(h_conv1_reshape, [-1, lstm1_timesteps, 24 * 24])
    W_lstm1 = weight_variable([2 * lstm1_num_hidden, 1536])
    h_conv1_reshape = tf.unstack(h_conv1_reshape, lstm1_timesteps, 1)
    lstm1_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm1_num_hidden, forget_bias=1.0)
    lstm1_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm1_num_hidden, forget_bias=1.0)
    lstm1_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm1_cell_fw, lstm1_cell_bw, h_conv1_reshape, dtype=tf.float32)
h_lstm1 = tf.matmul(lstm1_outputs[-1], W_lstm1)

# Dense Layer 1
W_dense1 = weight_variable([3 * 3 * 192, 1536])
b_dense1 = bias_variable([1536])
h_conv3 = tf.reshape(h_conv3, [-1, 3 * 3 * 192])
h_dense1 = tf.nn.relu(tf.matmul(h_conv3, W_dense1) + h_lstm1 + h_lstm2 + b_dense1)

# Dense Layer 2
W_dense2 = weight_variable([1536, 256])
b_dense2 = bias_variable([256])
h_dense2 = tf.nn.relu(tf.matmul(h_dense1, W_dense2) + b_dense2)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_dense2_drop = tf.nn.dropout(h_dense2, keep_prob)

# Dense Layer 3 with Softmax Output
W_dense3 = weight_variable([256, 10])
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
    data_location = './MNIST_LSTM_ConvNet'
    #saver.restore(sess, data_location)
    last_time = time.time()
    rate = 0.0001
    for i in range(100000):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={
                 x: batch[0], y_: batch[1], keep_prob: 0.5, training_rate: rate})
        if i % 10 == 0:
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={
                                 x: batch[0], y_: batch[1], keep_prob: 1.0, training_rate: rate})
            print('Step: %d, Accuracy: %.2f, Loss: %.5f, Speed: %.1f sec/10 steps' %
                  (i, acc, loss, time.time() - last_time))
            last_time = time.time()                
        if i % 1000 == 0 and i > 0:
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