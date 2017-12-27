import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 784])

W1 = weight_variable([784, 10])
b1 = bias_variable([10])

y = tf.matmul(x,W1) + b1
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(30000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 10 == 0:
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print('Step: %d Training Accuracy: %g' %(i, train_accuracy))
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

