import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from Layer import Layer

class Discriminator:

    # 判别器网络
    def DisNet():
        D_W1 = Layer.weight_var([784, 128], 'D_W1')
        D_b1 = Layer.bias_var([128], 'D_b1')
        D_W2 = Layer.weight_var([128, 1], 'D_W2')
        D_b2 = Layer.bias_var([1], 'D_b2')
        D = [D_W1, D_W2,D_b1, D_b2]
        return D

    #是真实mnist的概率
    def discriminator(D,x):
        D_h1 = tf.nn.relu(tf.matmul(x, D[0]) + D[2])
        D_logit = tf.matmul(D_h1, D[1]) + D[3]
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

