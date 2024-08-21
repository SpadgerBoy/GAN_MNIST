import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from Layer import Layer

class Generator:
        
    # 生成器网络
    def GenNet():
        G_W1 = Layer.weight_var([100, 128], 'G_W1')
        G_b1 = Layer.bias_var([128], 'G_B1')
        G_W2 = Layer.weight_var([128, 784], 'G_W2')
        G_b2 = Layer.bias_var([784], 'G_B2')
        G = [G_W1, G_W2, G_b1, G_b2]
        return G

    #先验空间到数据集的一个映射
    def generator(G,z):
        G_h1 = tf.nn.relu(tf.matmul(z, G[0]) + G[2])
        G_log_prob = tf.matmul(G_h1, G[1]) + G[3]
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob
