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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def rnn_layer(x, timesteps, num_hidden, weights, bias):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell_a = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_cell_b = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_cell_a, lstm_cell_b, x, dtype=tf.float32)
    return tf.nn.relu(tf.matmul(outputs[-1], weights) + bias)

def rnn_mix(x, timesteps, num_hidden, weights):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _ = tf.contrib.rnn.static_rnn(
        lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights)

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Main ConvLayer with max-pooling
with tf.variable_scope('ConvLayer'):
    W_conv = weight_variable([5, 5, 1, 64])
    b_conv = bias_variable([64])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)

with tf.variable_scope('ConvLayers'):
    # ConvLayer 1
    W_conv1 = weight_variable([3, 3, 64, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(h_pool, W_conv1) + b_conv1)

    # ConvLayer 2
    W_conv2 = weight_variable([3, 3, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # ConvLayer 3
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

    # ConvLayer 4
    W_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # ConvLayer 5
    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

# LSTM Branches Params
lstm_num_hidden = 64
lstm_timesteps = 64
lstm_num_output = 512

# LSTM Branch from ConvLayer 1
with tf.variable_scope('LSTM1'):
    h_conv1_reshape = tf.reshape(tf.transpose(h_conv1, [0, 3, 1, 2]), [-1, lstm_timesteps, 14 * 14])
    W_lstm1 = weight_variable([lstm_num_hidden * 2, lstm_num_output])
    b_lstm1 = bias_variable([lstm_num_output])
    h_lstm1 = rnn_layer(h_conv1_reshape, lstm_timesteps, lstm_num_hidden, W_lstm1, b_lstm1)

# LSTM Branch from ConvLayer 2
with tf.variable_scope('LSTM2'):
    h_conv2_reshape = tf.reshape(tf.transpose(h_conv1, [0, 3, 1, 2]), [-1, lstm_timesteps, 14 * 14])
    W_lstm2 = weight_variable([lstm_num_hidden * 2, lstm_num_output])
    b_lstm2 = bias_variable([lstm_num_output])
    h_lstm2 = rnn_layer(h_conv2_reshape, lstm_timesteps, lstm_num_hidden, W_lstm2, b_lstm2)

# LSTM Branch from ConvLayer 3
with tf.variable_scope('LSTM3'):
    h_conv3_reshape = tf.reshape(tf.transpose(h_conv3, [0, 3, 1, 2]), [-1, lstm_timesteps, 14 * 14])
    W_lstm3 = weight_variable([lstm_num_hidden * 2, lstm_num_output])
    b_lstm3 = bias_variable([lstm_num_output])
    h_lstm3 = rnn_layer(h_conv3_reshape, lstm_timesteps, lstm_num_hidden, W_lstm3, b_lstm3)

# LSTM Branch from ConvLayer 4
with tf.variable_scope('LSTM4'):
    h_conv4_reshape = tf.reshape(tf.transpose(h_conv4, [0, 3, 1, 2]), [-1, lstm_timesteps, 14 * 14])
    W_lstm4 = weight_variable([lstm_num_hidden * 2, lstm_num_output])
    b_lstm4 = bias_variable([lstm_num_output])
    h_lstm4 = rnn_layer(h_conv4_reshape, lstm_timesteps, lstm_num_hidden, W_lstm4, b_lstm4)

# LSTM Branch from ConvLayer 5
with tf.variable_scope('LSTM5'):
    h_conv5_reshape = tf.reshape(tf.transpose(h_conv5, [0, 3, 1, 2]), [-1, lstm_timesteps, 14 * 14])
    W_lstm5 = weight_variable([lstm_num_hidden * 2, lstm_num_output])
    b_lstm5 = bias_variable([lstm_num_output])
    h_lstm5 = rnn_layer(h_conv5_reshape, lstm_timesteps, lstm_num_hidden, W_lstm5, b_lstm5)

# LSTM Mixed
with tf.variable_scope('LSTM'):
    lstm_mix_num_hidden = 64
    b_mix = bias_variable([lstm_num_output])
    h_mix = tf.stack([h_lstm1, h_lstm2, h_lstm3, h_lstm4, h_lstm5])
    print(h_mix)
    h_mix_reshape = tf.transpose(h_mix, [1, 0, 2])
    print(h_mix_reshape)
    W_mix = weight_variable([lstm_mix_num_hidden, 4096])
    h_mix_out = rnn_mix(h_mix_reshape, 5, lstm_mix_num_hidden, W_mix)

# Dense Layer 1
W_dense1 = weight_variable([14 * 14 * 64, 4096])
b_dense1 = bias_variable([4096])
h_conv5 = tf.reshape(h_conv5, [-1, 14 * 14 * 64])
h_dense1 = tf.nn.relu(tf.matmul(h_conv5, W_dense1) + h_mix_out + b_dense1)

# Dense Layer 2
W_dense2 = weight_variable([4096, 1024])
b_dense2 = bias_variable([1024])
h_dense2 = tf.nn.relu(tf.matmul(h_dense1, W_dense2) + b_dense2)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_dense2_drop = tf.nn.dropout(h_dense2, keep_prob)

# Dense Layer 3 with Softmax Output
W_dense3 = weight_variable([1024, 10])
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
    data_location = './Mixed-LSTM_ConvNet-DATA/MNIST'
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