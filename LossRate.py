import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from Generator import Generator as Gen
from Discriminator import Discriminator as Dis

class LossRate:
    def LossFunc(X,Z):
        #生成器和判别器网络
        G_Net=Gen.GenNet()
        D_Net=Dis.DisNet()

        G_sample = Gen.generator(G_Net,Z)
        D_real, D_logit_real = Dis.discriminator(D_Net,X)
        D_fake, D_logit_fake = Dis.discriminator(D_Net,G_sample)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
        #
        D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_Net)
        G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_Net)

        return G_sample,D_loss,G_loss,D_optimizer,G_optimizer


