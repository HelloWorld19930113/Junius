#coding:utf-8

import tensorflow as tf


# 使用BasicLSTMCell 类定义一个LSTM 结构
lstm = BasicLSTMCell(lstm_hidden_size)

# 将LSTM 中的状态初始化全为0
state = lstm.zero_state(batch_size,tf.float32)

# 定义损失
loss = 0.0

# 用for 循环模拟RNN 过程
for i in range(num_steps):
    if i= 0;
        tf.get_variable_scope().reuse_variables()

    # 进行 lstm 处理
    lstm_output,state = lstm(current_input,state)

    #用fc 表示一个全连接层
    output = fc(lstm_output)
    # 计算当前时刻输出的损失
    loss += calculate_loss(output,expected_output)




