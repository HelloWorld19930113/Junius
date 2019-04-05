#coding:utf-8

import tensorflow as tf
import reader
import time
import numpy as np

# jungang.tjg
# 2019.04.05
# lstm 长短期记忆网络模型实现


class PTBModel(object):
    def __init__(self,is_training,config,data,name=None):
        print(">>> start run init ...")

        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data,
                                batch_size,
                                num_steps,name=name)
        self.size = config.hidden_size
        self.vocab_size = config.vocab_size

        #以LSTM 结构作为循环体，并且在训练时使用dropout
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.size,
                                                forget_bias=0.0,
                                                state_is_tuple=True)
        if is_training and config.keep_prob < 1.0:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,config.keep_prob)

        # 堆叠LSTM 单元 lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(config.num_layers)],
                                            state_is_tuple=True)

        # 初始化状态为0
        self.initial_state = cell.zero_state(batch_size,tf.float32)

        # 将输入单词转化为词向量
        embedding = tf.get_variable("embedding",[self.vocab_size,self.size],
                                    dtype=tf.float32)
        # 转化后的输入层维度为batch_size x num_steps x size
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)

        # 训练时对输入加上dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        
        # 定义输出列表,先将不同时刻LSTM结构的输出收集起来，再一起提供给softmax层
        # 通过一个全连层得到最终的输出
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                # 从输入数据获取当前时刻的输入并传入LSTM结构
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # concat()将输出的outputs展开成[batch_size,size*num_steps]的形状
        # 然后reshape()函数转为[batch_size*num_steps，size]的形状
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.size])

        weight = tf.get_variable("softmax_w", [self.size, self.vocab_size],
                                dtype=tf.float32)
        bias = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, weight) + bias

        # 定义损失函数，用序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],[tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        # 计算每个batch 的平均损失
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        self.learning_rate = tf.Variable(0.0,trainable=False)
        trainable_variables = tf.trainable_variables()

        # 更新梯度
        gradients = tf.gradients(cost, trainable_variables)

        # 通过clip_by_global_norm()控制梯度大小，防止发生梯度爆炸
        clipped_grads, _ = tf.clip_by_global_norm(gradients,config.max_grad_norm)

        # 使用SGD 进行梯度计算
        SGDOptimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # 定义训练算子
        self.train_op = SGDOptimizer.apply_gradients(zip(clipped_grads, trainable_variables),
                                                    global_step=tf.contrib.framework.get_or_create_global_step())
        # 更新学习率
        self.new_learning_rate = tf.placeholder(tf.float32, shape=[],name="new_learning_rate")
        self.learning_rate_update = tf.assign(self.learning_rate, self.new_learning_rate)


    # 定义学习率分配函数
    def assign_lr(self, session, lr_value):
        session.run(self.learning_rate_update, feed_dict={self.new_learning_rate: lr_value})

class Config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    total_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session, model, train_op=None, output_log=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if train_op is not None:
        fetches["train_op"] = train_op

    for step in range(model.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        result = session.run(fetches,feed_dict)

        cost = result["cost"]
        state = result["final_state"]

        costs += cost
        iters += model.num_steps

        if output_log and step % (model.epoch_size//10) == 10:
            print("step%.3f perplexity: %.3f speed: %.0f words/sec" %(step, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

# 读取数据集
train_data, valid_data, test_data, _ = reader.ptb_raw_data("../data/")

config = Config()
eval_config = Config()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    # 定义用于训练的循环神经网络模型
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            Model_train = PTBModel(is_training=True, config=config, data=train_data,
                            name="TrainModel")

    # 定义用于验证的循环神经网络模型
    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            Model_valid = PTBModel(is_training=False, config=config, data=valid_data,
                            name="ValidModel")

    # 定义用于测试的循环神经网络模型
    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            Model_test = PTBModel(is_training=False, config=eval_config,data=test_data,
                            name="TestModel")

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.total_epoch):
            # 确定学习率衰减，config.max_epoch代表了使用初始学习率的epoch
            # 在这几个epoch内lr_decay会是1
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            Model_train.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(Model_train.learning_rate)))

            # 在所有训练数据上训练循环神经网络模型
            train_perplexity = run_epoch(session, Model_train, train_op=Model_train.train_op,output_log=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            # 使用验证数据评测模型效果
            valid_perplexity = run_epoch(session, Model_valid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        print(">>> start run test data ...")
        # 最后使用测试数据测试模型的效果
        test_perplexity = run_epoch(session, Model_test)
        print("Test Perplexity: %.3f" % test_perplexity)


