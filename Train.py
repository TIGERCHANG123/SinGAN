import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, size_list, gp=0.1):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.size_list = size_list
        self.gp = gp

        self.grad_penalty = 0
    def get_loss(self, output):
        loss_list = []
        for i in range(len(output)):
            loss_list.append(tf.reduce_mean(output[i]))
        return loss_list
    def train_step(self, image):
        z_list = []
        image1 = image
        image2 = image
        for i in range(self.size_list):
            z_list.append(tf.zeros(shape=[image1.shape[0], self.size_list[i], self.size_list[i], 3]))
        z_list[0] = tf.convert_to_tensor(np.random.randn(z_list[0].shape))
        with tf.GradientTape() as tape:
            mse_image_list = self.generator(image1, z_list, training=True)
            g_rec = tf.keras.losses.MSE(mse_image_list[-1], image2)

            rmse_list = [1.0]
            for i, mse_img in enumerate(mse_image_list):
                if i != 0:
                    resize_img = tf.image.resize(image2, mse_img.shape)
                    rmse = tf.math.sqrt(tf.keras.losses.MSE(mse_image_list[i], resize_img))
                    rmse_list.append(rmse)
            z_list = [rmse_list[i] * tf.convert_to_tensor(np.random.randn(z_list[i].shape)) for i in range(len(self.size_list))]

            fake_image_list = self.generator(image1, z_list, training=True)
            real_output_list, fake_output_list = self.discriminator(image2, fake_image_list, training=False)

            fake_loss_list = self.get_loss(fake_output_list)
            real_loss_list = self.get_loss(real_output_list)
            # gp calculation
            grad_penalty_list = []
            for i, fake_img in enumerate(fake_image_list):
                resize_img = tf.image.resize(image2, fake_img.size)
                rate = np.random.rand()
                mixed_pic = rate * resize_img + (1 - rate) * fake_img
                with tf.GradientTape() as mixed_tape:
                    mixed_tape.watch(mixed_pic)
                    mixed_output = self.discriminator(mixed_pic)
                grad_mixed = mixed_tape.gradient(mixed_output, mixed_pic)
                norm_grad_mixed = tf.sqrt(tf.reduce_sum(tf.square(grad_mixed), axis=[1, 2, 3]))
                grad_penalty_list.append(tf.reduce_mean(tf.square(norm_grad_mixed - 1)))
            disc_loss_list = []
            for i in range(len(fake_loss_list)):
                disc_loss_list.append(fake_loss_list[i] - real_loss_list[i] + self.gp * grad_penalty_list[i])

        self.gen_loss(-np.mean(fake_loss_list))
        self.disc_loss(-np.mean(real_loss_list))
        for i, loss in enumerate(fake_loss_list):
            trainable_variables = [v for v in self.generator.trainable_variables if 'generator_{}'.format(i) in v.name]
            gradients_of_generator = tape.gradient(loss + g_rec, trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, trainable_variables))
            trainable_variables = [v for v in self.discriminator.trainable_variables if 'discriminator_{}'.format(i) in v.name]
            gradients_of_discriminator = tape.gradient(disc_loss_list[i], trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, trainable_variables))

    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, images) in enumerate(self.train_dataset):
            self.train_step(images)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}'.format(epoch, self.gen_loss.result(), self.disc_loss.result()))