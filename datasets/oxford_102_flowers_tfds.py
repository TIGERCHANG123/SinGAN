import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow_datasets as tfds

class oxford_102_flowers_dataset():
    def __init__(self, root, batch_size):
        self.file_path = root + 'datasets/tensorflow_datasets'
        self.image_width = 128
        self.batch_size = batch_size
        self.name = 'oxford-102-flowers'
        self.dataset, meta = tfds.load(name='oxford_flowers102', with_info=True, as_supervised=True)
    def parse(self, x):
        image = x['image']
        y = x['label']
        x = image
        x = tf.cast(x, tf.float32)
        x = x / 255 * 2 - 1
        return x, y

    def get_train_dataset(self):
        #         return self.dataset
        train = self.dataset['train'].map(self.parse).shuffle(1000).batch(self.batch_size)
        test = self.dataset['test'].map(self.parse).shuffle(1000).batch(self.batch_size)
        valid = self.dataset['validation'].map(self.parse).shuffle(1000).batch(self.batch_size)
        return train, test, valid

class noise_generator():
    def __init__(self, noise_dim, digit_dim, batch_size):
        self.noise_dim = noise_dim
        self.digit_dim = digit_dim
        self.batch_size = batch_size
    def get_noise(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.digit_dim * [float(1.0 / self.digit_dim)],size=[self.batch_size])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        return noise, auxi_dict

    def get_fixed_noise(self, num):
        noise = tf.random.normal([1, self.noise_dim])
        noise = tf.cast(noise, tf.float32)

        auxi_dict = np.array([num])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.one_hot(auxi_dict, depth=self.digit_dim)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        return noise, auxi_dict