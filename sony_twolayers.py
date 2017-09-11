import tensorflow as tf
import numpy as np

def batch(x,y,n):
    index = np.random.choice(28094, n, replace=False)
    return [[x[i] for i in index],[y[i] for i in index]]

def computer_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

train_x = np.loadtxt(open("train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("test_y.csv","rb"),delimiter=",",skiprows=1)

xs = tf.placeholder(tf.float32, [None, 103])
ys = tf.placeholder(tf.float32, [None, 7])

Weights = tf.Variable(tf.random_normal([103, 10]))
biases = tf.Variable(tf.zeros([1, 10]) + 0.1)
Wx_plus_b = tf.matmul(xs, Weights) + biases
layer1_out = tf.nn.relu(Wx_plus_b)

Weights_2 = tf.Variable(tf.random_normal([10,7]))
biases_2 = tf.Variable(tf.zeros([1, 7]) + 0.1)
Wx_plus_b_2 = tf.matmul(layer1_out, Weights_2) + biases_2
prediction = tf.nn.softmax(Wx_plus_b_2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(0, 20000):
    [batch_x,batch_y] = batch(train_x,train_y,300)
    sess.run(train_step, feed_dict={xs:batch_x,ys:batch_y})
    if step % 200 == 0:
        #print(sess.run(Weights))
        print(computer_accuracy(test_x, test_y))
        #print(step, sess.run(biases))
        #pass

#print(sess.run(Weights))