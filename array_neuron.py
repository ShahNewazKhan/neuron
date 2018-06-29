import tensorflow as tf

#https://www.oreilly.com/learning/hello-tensorflow

x = tf.constant([0.0,2.0,3.0,4.0,12.0], name='inputs')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
y_ = tf.constant([0.0,1.0,1.0,0.0,2.0], name='correct_values')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x,y,y_,w,loss]:
    tf.summary.tensor_summary(value.op.name, value)

summaries = tf.summary.merge_all()

sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
    
    print('before step {}, y is {}'.format(i, sess.run(y)))
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)
