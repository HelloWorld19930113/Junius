#coding:utf-8

import tensorflow as tf
import input_data

#导入mnist 数据集
mnist = input_data.read_data_sets("mnist_data/",one_hot = True)


# 查看训练数据的大小
print(mnist.train.images.shape)  # (55000, 784)
print(mnist.train.labels.shape)  # (55000, 10)


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义权重，bias
weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, weight) + bias

#定义交叉熵损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

#梯度下降最小化损失函数
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Compute loss.
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(y_, 1), logits=y)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 准备验证数据
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(30000):
        if i % 1000 == 0:
            # 计算滑动平均模型在验证数据上的结果。
            # 为了能得到百分数输出，需要将得到的validate_accuracy扩大100倍
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            validate_loss = sess.run(loss, feed_dict=validate_feed)

            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%% loss: %f" % (i, validate_accuracy * 100,validate_loss))

        xs, ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})

    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After 30000 trainging step(s) ,test accuracy using average"
          " model is %g%%" % (test_accuracy * 100))




