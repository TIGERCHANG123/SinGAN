import tensorflow as tf
import cv2

class photo_dataset():
    def __init__(self, root, batch_size, total_images):
        self.file_path = root + '/datasets/one_photo'
        self.batch_size = batch_size
        self.total_images = total_images
        self.file_name = self.file_path + '/balloons.png'
        self.name = 'one_photo'
    def generator(self):
        for i in range(self.total_images):
            yield cv2.imread(self.file_name, 1)
    def parse(self, x):
        x = tf.cast(x, tf.float32)
        x = x/255 * 2 - 1
        return x
    def get_train_dataset(self):
        train = tf.data.Dataset.from_generator(self.generator, output_types=tf.int64)
        train = train.map(self.parse).shuffle(1000).batch(self.batch_size)
        return train
