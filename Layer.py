import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class Layer(object):

    def weight_var(shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.keras.initializers.glorot_normal())

    def bias_var(shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


