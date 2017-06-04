# coding: utf-8
import tensorflow as tf

# Khởi tạo các biến
with tf.name_scope("a_simple_neuron") as scope:
	x = tf.constant(1.0)
	w = tf.Variable(0.8, name="weight")
	b = tf.Variable(0.5, name="bias")
	y = tf.add(tf.mul(w, x), b, name="output")

y_correct = tf.constant(1.6, name="expected_output")

with tf.name_scope("cost_function") as scope:
	cost = (y_correct - y)**2

with tf.name_scope("train_step") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

sess = tf.Session()
file_writer = tf.summary.FileWriter("Lab2", sess.graph)
sess.run(tf.global_variables_initializer())

# Tạo các đồ thị thể hiện sự biến thiên của weight, bias, cost, và output
summary_weight = tf.summary.scalar('weight', w)
summary_bias = tf.summary.scalar('bias', b)
summary_cost = tf.summary.scalar('cost', cost)
summary_y = tf.summary.scalar('output', y)

# Thực hiện quá trình học
for i in range(100):
	# Tính và log các giá trị hiện tại của weight, bias, cost, và output
	summaries = sess.run([summary_weight, summary_bias, summary_cost, summary_y])
	for summary in summaries:
		file_writer.add_summary(summary, i)
		
	# Gọi bước học tiếp theo
	sess.run(train_step)
sess.close()