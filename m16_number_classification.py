#手写体数字识别

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#数据集,没有的话会自动下载
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs,v_ys):
    #计算准确度
    global prediction          #tf.argmax(input, axis=None, name=None, dimension=None)此函数是对矩阵按行或列计算最大值input：输入Tensor,axis：0表示按列，1表示按行
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})  #tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#placeholder
xs = tf.placeholder(tf.float32,[None,784])#28x28
ys = tf.placeholder(tf.float32,[None,10])

#输出层
prediction = add_layer(xs,784,10,activation_function = tf.nn.softmax)

#误差分析(softmax+cross_entropy) 交叉熵+softmax是非常好的分类方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    #批量(batch)从数据集中提取,每次100个,每次学习整个数据集会很慢
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        #测试时用的是专门的测试集,不是训练集
        print(compute_accuracy(mnist.test.images,mnist.test.labels))



