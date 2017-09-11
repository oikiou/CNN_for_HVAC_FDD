import tensorflow as tf
import numpy as np

def batch(x,y,n):
    index = np.random.choice(21894, n, replace=False)
    return [[x[i] for i in index],[y[i] for i in index]]

def computer_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

train_x = np.loadtxt(open("night_train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("night_train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("night_test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("night_test_y.csv","rb"),delimiter=",",skiprows=1)

xs = tf.placeholder(tf.float32, [None, 13])
ys = tf.placeholder(tf.float32, [None, 7])

Weights = tf.Variable(tf.random_normal([13, 20]))
biases = tf.Variable(tf.zeros([1, 20]) + 0.01)
Wx_plus_b = tf.matmul(xs, Weights) + biases
layer1_out = tf.nn.relu(Wx_plus_b)

Weights_2 = tf.Variable(tf.random_normal([20,20]))
biases_2 = tf.Variable(tf.zeros([1, 20]) + 0.01)
Wx_plus_b_2 = tf.matmul(layer1_out, Weights_2) + biases_2
layer2_out = tf.nn.relu(Wx_plus_b_2)

Weights_3 = tf.Variable(tf.random_normal([20,7]))
biases_3 = tf.Variable(tf.zeros([1, 7]) + 0.01)
Wx_plus_b_3 = tf.matmul(layer2_out, Weights_3) + biases_3
prediction = tf.nn.softmax(Wx_plus_b_3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0002).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(0, 50000):
    [batch_x,batch_y] = batch(train_x,train_y,500)
    sess.run(train_step, feed_dict={xs:batch_x,ys:batch_y})
    if step % 100 == 0:
        #print(sess.run(Weights))
        print(computer_accuracy(test_x, test_y), computer_accuracy(train_x, train_y))
        #print(step, sess.run(biases))
        #pass

#print(sess.run(Weights))
test_y_output = sess.run(prediction, feed_dict={xs: test_x})
train_y_output = sess.run(prediction, feed_dict={xs: train_x})
np.savetxt('result_prediction_1.csv', train_y_output, delimiter = ',', fmt="%.3f")
np.savetxt('result_prediction_2.csv', test_y_output, delimiter = ',', fmt="%.3f")