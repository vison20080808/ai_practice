

# 【AI实战】快速掌握TensorFlow（二）：计算图、会话
# https://my.oschina.net/u/876354/blog/1930490


# 使用图（Graph）来表示计算任务，由节点和边组成。TensorFlow由前端负责构建计算图，后端负责执行计算图。
# 为了执行图的计算，图必须在会话（Session）里面启动，会话将图的操作分发到CPU、GPU等设备上执行。

import tensorflow as tf

# 1、图（Graph）
# 已经有一个默认图 (default graph)，如果没有创建新的计算图，则默认情况下是在这个default graph里面创建节点和边。
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')
result = tf.add(a, b)
# 现在默认图就有了三个节点，两个constant()，和一个add()。


# 2、会话（Session）
# tf.Session(graph=g1) 指定某个计算图g1
# （1）创建一个会话
sess = tf.Session()
result_val = sess.run(result)
print(result_val)
# 任务完成, 关闭会话.
sess.close()

# （2） 创建一个会话
with tf.Session() as sess:
    result_val = sess.run(result)
    print(result_val)

# （3）创建一个默认的会话
sess = tf.Session()
with sess.as_default():
    result_val = result.eval()  # 当指定默认会话后，可以通过tf.Tensor.eval函数来计算一个张量的取值。
    print(result_val)

# （4）创建一个交互式会话
sess = tf.InteractiveSession()  # 该函数会自动将生成的会话注册为默认会话，使用 tf.Tensor.eval()代替 Session.run()
result_val = result.eval()
print(result_val)
sess.close()


# 3、构建多个计算图
# 在TensorFlow中可以构建多个计算图，计算图之间的张量和运算是不会共享的，通过这种方式，可以在同个项目中构建多个网络模型，而相互之间不会受影响。
g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量'v',并设置初始值为0。
    v = tf.get_variable('v', initializer=tf.zeros_initializer()(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量'v',并设置初始值微1。
    v = tf.get_variable('v', initializer=tf.ones_initializer()(shape=[1]))

# 在计算图g1中读取变量'v'的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))


# 在计算图g2中读取变量'v'的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))
        # 输出结果[1.]。


# 4、指定运行设备
# （1）在图中指定运行设备
g = tf.Graph()
with g.device('/cpu:0'):  # "/gpu:0"  "/gpu:1"
    result = tf.add(a, b)

# （2）在会话中指定运行设备
with tf.Session() as sess:
    with tf.device('/cpu:0'):  # "/gpu:0"  "/gpu:1"
        result = tf.add(a, b)
        print(sess.run(result))






