from __future__ import print_function
import tensorflow as tf
import numpy as np

train_x = np.loadtxt(open("cnn_train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("cnn_train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("cnn_test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("cnn_test_y.csv","rb"),delimiter=",",skiprows=1)
test_x_bems = np.loadtxt(open("cnn_test_x_bems.csv","rb"),delimiter=",",skiprows=1)

def batch(x,y,n):
    index = np.random.choice(377, n, replace=False)
    return [[x[i] for i in index],[y[i] for i in index]]

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 15552]) # 90x90
ys = tf.placeholder(tf.float32, [None, 7])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 144, 108, 1])
# print(x_image.shape)

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,16]) #patch 5x5, insize 1, outsize 16
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #90x90x16
h_pool1 = max_pool_3x3(h_conv1) #30x30x16

## conv2 layer ##
W_conv2 = weight_variable([5,5,16,32]) #patch 5x5, insize 16, outsize 32
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #30x30x32
h_pool2 = max_pool_3x3(h_conv2) #10x10x32

## fully connect 1 layer ##

W_f1 = weight_variable([16*12*32, 1024])
b_f1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 16*12*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_f1) + b_f1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fully connect 2 layer ##
W_f2 = weight_variable([1024, 7])
b_f2 = bias_variable([7])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_f2) + b_f2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(2500):
    [batch_x, batch_y] = batch(train_x, train_y, 20)
    sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(test_x, test_y),compute_accuracy(train_x, train_y))

test_y_output = sess.run(prediction, feed_dict={xs: test_x, keep_prob: 1})
train_y_output = sess.run(prediction, feed_dict = {xs:train_x, keep_prob: 1})
test_bems = sess.run(prediction, feed_dict = {xs:test_x_bems, keep_prob: 1})
np.savetxt('test_result_prediction.csv', test_y_output, delimiter = ',', fmt="%.3f")
np.savetxt('train_result_prediction.csv', train_y_output, delimiter = ',', fmt="%.3f")
np.savetxt('result_bems.csv', test_bems, delimiter = ',', fmt="%.3f")