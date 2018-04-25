# add-layers.py
#
# to run
# python add-layers.py
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])

conv1_input = tf.reshape(x, [-1, 28, 28, 1])
conv1 = tf.layers.conv2d(inputs=conv1_input,
                         filters=3,
                         kernel_size=[5, 5],
                         padding='valid')

conv1 = tf.reshape(conv1, [-1, 1728])

print(conv1)

W1 = tf.get_variable("weights1", shape=[1728, 350],
                    initializer=tf.glorot_uniform_initializer())

b1 = tf.get_variable("bias1", shape=[350],
                    initializer=tf.constant_initializer(0.1))

W2 = tf.get_variable("weights2", shape=[350, 175],
                    initializer=tf.glorot_uniform_initializer())

b2 = tf.get_variable("bias2", shape=[175],
                    initializer=tf.constant_initializer(0.1))

W3 = tf.get_variable("weights3", shape=[175, 10],
                    initializer=tf.glorot_uniform_initializer())

b3 = tf.get_variable("bias3", shape=[10],
                    initializer=tf.constant_initializer(0.1))


out1 = tf.nn.relu(tf.matmul(conv1, W1) + b1)
out2 = tf.nn.relu(tf.matmul(out1, W2) + b2)
y = tf.nn.relu(tf.matmul(out2, W3) + b3)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# We'll use this to make predictions with our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for step in range(50):
    #print("training step: ", step)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if step % 10 == 0:
        print("model accuracy: ")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))

print("final model accuracy: ")
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))