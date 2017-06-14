import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.name_scope("Input") as scope:
	x = tf.placeholder(tf.float32, [None, 784], name="x-input")               
	y_correct = tf.placeholder(tf.float32, [None, 10], name="y-correct_label")
	
with tf.name_scope("Weight") as scope:
	W = tf.Variable(tf.zeros([784, 10]), name="weight")                       
	
with tf.name_scope("Bias") as scope:
	b = tf.Variable(tf.zeros(10), name="bias")

with tf.name_scope("Softmax") as scope:
	y = tf.nn.softmax(tf.matmul(x, W) + b, name="softmax")                # * 
	
with tf.name_scope("Cross_Entropy") as scope:                             # *
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_correct * tf.log(y), 
												  reduction_indices=[1]), 
								   name="cross_entropy")
# *: https://www.tensorflow.org/get_started/mnist/beginners#training.
with tf.name_scope('Train') as scope:
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
with tf.name_scope("Accuracy"):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_correct,1), 
			name = "correct_prediction")
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the model
sess = tf.InteractiveSession()
file_writer = tf.summary.FileWriter("DNN_0_hidden_layer", sess.graph)
# create a summary for our cost and accuracy
tf.summary.scalar("cost_summary", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
# merge all summaries into a single operation 
# which we can execute in a session 
summary_step = tf.summary.merge_all()

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    # batch_size = 100
    _, summary = sess.run([train_step, summary_step], 
                          feed_dict={x: batch_xs, y_correct: batch_ys})
    # logging
    file_writer.add_summary(summary, i)
# Test
print("Accuracy: {}".format(accuracy.eval(feed_dict={x: mnist.test.images, 
                                             y_correct: mnist.test.labels})))