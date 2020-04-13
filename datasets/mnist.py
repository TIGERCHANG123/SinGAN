from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf

class mnist_dataset():
    def __init__(self, root, noise_dim):
        file_path = root + '/datasets/tensorflow_datasets'
        mnist, meta = tfds.load('mnist', data_dir=file_path, download=False, as_supervised=True, with_info=True)
        print(meta)
        self.train_dataset=mnist['train']
        self.noise_dim = noise_dim
        self.name = 'mnist'
        return
    def parse(self, x, y):
        x=tf.cast(x, tf.float32)
        # x = tf.expand_dims(x, -1)
        x=x/255*2 - 1
        y = tf.cast(y, tf.int64)
        return x, y
    def get_train_dataset(self):
        train_dataset = self.train_dataset.map(self.parse).shuffle(60000).batch(128)
        return train_dataset

class noise_generator():
    def __init__(self, noise_dim, digit_dim, batch_size):
        self.digit_dim = digit_dim
        self.batch_size = batch_size
        self.noise_dim = noise_dim
    def get_noise(self, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.digit_dim * [float(1.0 / self.digit_dim)],size=[batch_size])
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


