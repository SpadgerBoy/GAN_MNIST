import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from tensorflow.examples.tutorials.mnist import input_data
tf.disable_eager_execution()

from Generator import Generator as Gen
from Discriminator import Discriminator as Dis
from Draw import Draw 
from LossRate import LossRate as LR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.InteractiveSession()

mb_size = 128   #数据集大小
Z_dim = 100     #生成器的初始维度

#加载数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
#获取损失率及最优点
G_sample,D_loss,G_loss,D_optimizer,G_optimizer=LR.LossFunc(X,Z)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

if __name__ == '__main__':

    #将生成的图片放入当前路径下的out文件夹
    if not os.path.exists('out/'):
        os.makedirs('out/')
    #将计算出的loss放入log文件夹
    if not os.path.exists('log/'):
        os.makedirs('log/')
    file_D='log/D_loss.log'
    file_G='log/G_loss.log'
    f_D=open(file_D,'w+')
    f_G=open(file_G,'w+')

    # 初始化
    sess.run(tf.global_variables_initializer())

    i = 0
    for it in range(100000):

        X_mb, _ = mnist.train.next_batch(mb_size)
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(9, Z_dim)})  
            fig = Draw.plot(samples)    #画图，每张图9*28*28
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

            f_D.write(str(D_loss_curr)+'\n')
            f_G.write(str(G_loss_curr)+'\n')
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}\n'.format(G_loss_curr))

    f_D.close()
    f_G.close()




