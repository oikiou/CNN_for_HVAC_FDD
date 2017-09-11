import tensorflow as tf
import numpy as np

def batch(x,y,n):
    index = np.random.choice(783, n, replace=False)
    return [[x[i] for i in index],[y[i] for i in index]]

def computer_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
'''
train_x = np.loadtxt(open("sony_train_stand_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("sony_train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("sony_test_stand_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("sony_test_y.csv","rb"),delimiter=",",skiprows=1)


train_x = np.loadtxt(open("new_train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("new_train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("new_test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("new_test_y.csv","rb"),delimiter=",",skiprows=1)

train_x = np.loadtxt(open("train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("test_y.csv","rb"),delimiter=",",skiprows=1)

train_x = np.loadtxt(open("train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("test_y.csv","rb"),delimiter=",",skiprows=1)
'''

train_x = np.loadtxt(open("cnn_90_16_7_7_train_x.csv","rb"),delimiter=",",skiprows=1)
train_y = np.loadtxt(open("cnn_90_16_7_7_train_y.csv","rb"),delimiter=",",skiprows=1)
test_x = np.loadtxt(open("cnn_90_16_7_7_test_x.csv","rb"),delimiter=",",skiprows=1)
test_y = np.loadtxt(open("cnn_90_16_7_7_test_y.csv","rb"),delimiter=",",skiprows=1)

xs = tf.placeholder(tf.float32, [None, 8100])
ys = tf.placeholder(tf.float32, [None, 7])

Weights = tf.Variable(tf.random_normal([8100, 7]))
biases = tf.Variable(tf.zeros([1, 7]) + 0.1)
Wx_plus_b = tf.matmul(xs, Weights) + biases
prediction = tf.nn.softmax(Wx_plus_b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(0, 100):
    [batch_x,batch_y] = batch(train_x,train_y,30)

    sess.run(train_step, feed_dict={xs:batch_x,ys:batch_y})
    if step % 5 == 0:
        #print(sess.run(Weights))
        #print(computer_accuracy(train_x,train_y),computer_accuracy(test_x, test_y))
        print(computer_accuracy(test_x, test_y))
        #print(step, sess.run(biases))
        #pass

#weight_output = sess.run(Weights)
#test_y_output = sess.run(prediction, feed_dict={xs: test_x})
#test_y_output_index = np.array(tf.argmax(test_y_output,1))
#np.savetxt('result_prediction_2.csv', test_y_output, delimiter = ',', fmt="%.3f")
#np.savetxt('weight_output.csv', weight_output, delimiter = ',', fmt="%.3f")
#np.savetxt('test_y_output_index.csv', test_y_output_index, delimiter = ',', fmt="%.3f")
#print(test_y_output_index)