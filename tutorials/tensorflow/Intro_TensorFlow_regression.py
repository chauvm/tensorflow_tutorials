# coding: utf-8
import tensorflow as tf

# Khởi tạo các biến
with tf.name_scope("a_simple_neuron") as scope:
	# Ma trận input ban đầu là
	# ([[2104, 5, 1, 45],
	# [1416, 3, 2, 40],
	# [1534, 3, 2, 30],
	# [852, 2, 1, 36]]]
	# được normalized bằng cách chia mỗi phần tử trong một cột
	# với norm của vector của cột đó
	# ví dụ norm của cột đầu tiên [2104, 1416, 1534, 852] là
	# norm = math.sqrt(2104**2 + 1416**2 + 1534**2 + 852**2) = 3083.97
	# do vậy phần tử 2104 sẽ trở thành 2014/3083.97 = 0.68
	# và 1416 thành 1416/3083.97 = 0.46
	x = tf.constant([[ 0.68223532,  0.72932496,  0.31622777,  0.58981215],
				   [ 0.45914696,  0.43759497,  0.63245553,  0.52427747],
				   [ 0.49740921,  0.43759497,  0.63245553,  0.3932081 ],
				   [ 0.27626639,  0.29172998,  0.31622777,  0.47184972]], dtype=tf.float32)

	# W là một ma trận 2 chiều, kích thước 4x1, 
	# với giá trị ban đầu của tất cả các phần tử là 0 
	W = tf.Variable(tf.zeros([4, 1]), dtype=tf.float32, name="weight")

	b = tf.Variable(0.5, name="bias")

	# Lưu ý dùng tf.matmul để nhân 2 ma trận, và x để trước W
	y = tf.add(tf.matmul(x, W), b, name="output")

# y_correct là một ma trận 4x1
y_correct = tf.constant([[460], [232], [315], [178]], dtype=tf.float32, 
						name="expected_output")

with tf.name_scope("cost_function") as scope:
	cost = tf.reduce_mean(tf.square(y_correct - y))

with tf.name_scope("train_step") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

sess = tf.Session()
file_writer = tf.summary.FileWriter("Lab2", sess.graph)
sess.run(tf.global_variables_initializer())

# Tạo các đồ thị thể hiện sự biến thiên của weight, bias, cost, và output
summary_bias = tf.summary.scalar('bias', b)
summary_cost = tf.summary.scalar('cost', cost)

# Thực hiện quá trình học
for i in range(100):
	# Tính và log các giá trị hiện tại của weight, bias, cost, và output
	summaries = sess.run([summary_weight, summary_bias, summary_cost, summary_y])
	for summary in summaries:
		file_writer.add_summary(summary, i)
		
	# Gọi bước học tiếp theo
	sess.run(train_step)
print("Weight: {}".format(sess.run(W)))
print("Bias: {}".format(sess.run(b)))
sess.close()
""" Kết quả
Weight: [[ 330.91555786]
 [ 333.26712036]
 [ -67.61777496]
 [ -17.64081955]]
Bias: 19.997024536132812
"""