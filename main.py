# -*- coding:utf-8 -*-
import os
import getopt
import sys
import tensorflow as tf
from SinGAN import get_gan
from show_pic import draw
import fid
from Train import train_one_epoch
from datasets.one_photo import photo_dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np


ubuntu_root='/home/tigerc'
windows_root='D:/Automatic/SRTP/GAN'
root = '/content/drive/My Drive'
root = windows_root
temp_root = root+'/temp'
# dataset_root = '/content'
dataset_root = root

def main(continue_train, train_time, train_epoch):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    noise_dim = 100
    batch_size = 3

    img_size_max = 250
    img_size_min = 25
    scale_factor = 4 / 3
    tmp_scale = img_size_max / img_size_min
    num_scale = int(np.round(np.log(tmp_scale) / np.log(scale_factor)))
    size_list = [int(img_size_min * scale_factor ** i) for i in range(num_scale + 1)]

    generator_model, discriminator_model, model_name = get_gan()
    dataset = photo_dataset(dataset_root,batch_size=1, total_images=2000)
    model_dataset = model_name + '-' + dataset.name

    train_dataset = dataset.get_train_dataset()
    pic = draw(10, temp_root, model_dataset, train_time=train_time)
    generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)

    checkpoint_path = temp_root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(genetator_optimizers=generator_optimizer, discriminator_optimizer=discriminator_optimizer ,
                               generator=generator_model, discriminator=discriminator_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint and continue_train:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')

    train = train_one_epoch(model=[generator_model, discriminator_model], train_dataset=train_dataset,
              optimizers=[generator_optimizer, discriminator_optimizer], metrics=[gen_loss, disc_loss], size_list=size_list, gp=0.1)

    for epoch in range(train_epoch):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic(generator_model, 8, noise_dim, epoch)
    pic.show_created_pic(generator_model, 8, noise_dim)

    # # fid score
    # gen = generator_model
    # noise = noise_generator(noise_dim, 10, batch_size, dataset.total_pic_num//batch_size)()
    # real_images = dataset.get_train_dataset()
    # fd = fid.FrechetInceptionDistance(gen, (-1, 1), [128, 128, 3])
    # gan_fid, gan_is = fd(iter(real_images), noise, batch_size=batch_size, num_batches_real=dataset.total_pic_num//batch_size)
    # print('fid score: {}, inception score: {}'.format(gan_fid, gan_is))

    return
if __name__ == '__main__':
    continue_train = False
    train_time = 0
    epoch = 500
    try:
        opts, args = getopt.getopt(sys.argv[1:], '-c-t:-e:', ['continue', 'time=', 'epoch='])
        for op, value in opts:
            print(op, value)
            if op in ('-c', '--continue'):
                continue_train = True
            elif op in ('-t', '--time'):
                train_time = int(value)
            elif op in ('-e', '--epoch'):
                epoch = int(value)
    except:
        print('wrong input!')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main(continue_train=continue_train, train_time=train_time, train_epoch=epoch)