import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.name_scope("Input") as scope:
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")               # None means a dimension can be of any length
    y_correct = tf.placeholder(tf.float32, [None, 10], name="y-correct_label")
    
with tf.name_scope("Weight") as scope:
    W = tf.Variable(tf.zeros([784, 10]), name="weight")                       # weights
    
with tf.name_scope("Bias") as scope:
    b = tf.Variable(tf.zeros(10), name="bias")                                # bias

with tf.name_scope("Softmax") as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b, name="softmax")

with tf.name_scope("Cross_Entropy") as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_correct * tf.log(y), reduction_indices=[1]), name="cross_entropy")

with tf.name_scope('Train') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_correct,1), name = "correct_prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar("cost_summary", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single operation
summary_step = tf.summary.merge_all()

# Launch the model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

file_writer = tf.summary.FileWriter("digit_classification_1_graphs", sess.graph)

# Print accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_correct: mnist.test.labels}))


