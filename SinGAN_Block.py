import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import *

class Conv_BN_LeakyRelu(tf.keras.Model):
    def __init__(self, filters, strides, padding, name):
        super(Conv_BN_LeakyRelu, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size=3, strides=strides, padding=padding, use_bias=False,
                                           kernel_initializer=RandomNormal(stddev=0.02), name=name+'_conv')
        self.bn = layers.BatchNormalization(momentum=0.9, name=name+'_bn')
        self.relu = tf.keras.layers.LeakyReLU(0.2, name=name+'_relu')
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class generator_First(tf.keras.Model):
  def __init__(self, nf, name):
    super(generator_First, self).__init__()
    self.conv_list = [
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu1'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu1'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu2'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu3'),
    ]
    self.out = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False,
                             kernel_initializer=RandomNormal(stddev=0.02), name=name + '_output')
    self.actv = tf.keras.layers.Activation('tanh', name=name+'_tanh')
  def call(self, x):
    for i in range(len(self.conv_list)):
        x = self.conv_list[i](x)
    x = self.out(x)
    x = self.actv(x)
    return x

class generator_Middle(tf.keras.Model):
  def __init__(self, nf, name):
    super(generator_Middle, self).__init__()
    self.conv_list = [
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu1'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu1'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu2'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name + '_conv_bn_relu3'),
    ]
    self.out = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False,
                             kernel_initializer=RandomNormal(stddev=0.02), name=name + '_output')
    self.actv = tf.keras.activations.tanh
  def call(self, x0, z0):
    x = x0 + z0
    for i in range(len(self.conv_list)):
        x = self.conv_list[i](x)
    x = self.out(x)
    x = self.actv(x)
    x = x + x0
    return x

class discriminator_block(tf.keras.Model):
  def __init__(self, nf, name):
    super(discriminator_block, self).__init__()
    self.conv = layers.Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', use_bias=False,
                              kernel_initializer=RandomNormal(stddev=0.02), name=name+'_conv')
    self.actv = keras.layers.LeakyReLU(0.2, name=name+'_actv')
    self.conv_list = [
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name+'_conv_bn_relu1'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name+'_conv_bn_relu2'),
        Conv_BN_LeakyRelu(nf, 1, 'same', name=name+'_conv_bn_relu3'),
    ]
    self.out = layers.Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', use_bias=False,
                              kernel_initializer=RandomNormal(stddev=0.02), name=name+'_output')
  def call(self, x):
      x = self.conv(x)
      x = self.actv(x)
      for i in range(len(self.conv_list)):
          x = self.conv_list[i](x)
      x = self.out(x)
      return x

