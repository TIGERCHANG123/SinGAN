from SinGAN_Block import *

class generator_model(tf.keras.Model):
  def __init__(self):
    super(generator_model, self).__init__()
    nf = 128
    self.generator_first = generator_First(nf, name='generator_0')
    self.generator_middle = [
      generator_Middle(nf, name='generator_1'),
      generator_Middle(nf, name='generator_2'),
      generator_Middle(nf, name='generator_3'),
      generator_Middle(2 * nf, name='generator_4'),
      generator_Middle(2 * nf, name='generator_5'),
      generator_Middle(2 * nf, name='generator_6'),
      generator_Middle(2 * nf, name='generator_7'),
    ]
  def call(self, img=None, z_list=None):
    x_list = []
    for i in range(len(z_list)):
      if i == 0:
        x = self.generator_first(z_list[0])
        x_list.append(x)
      else:
        x = tf.image.resize(img, z_list[i].shape)
        x = self.generator_middle[i-1](x, z_list[i])
        x_list.append(x)
    return x_list

class discriminator_model(tf.keras.Model):
  def __init__(self):
    super(discriminator_model, self).__init__()
    nf = 128
    self.discriminators = [
      discriminator(nf=nf,name='discriminator_0'),
      discriminator(nf=nf, name='discriminator_1'),
      discriminator(nf=nf, name='discriminator_2'),
      discriminator(nf=nf, name='discriminator_3'),
      discriminator(nf=2 * nf, name='discriminator_4'),
      discriminator(nf=2 * nf, name='discriminator_5'),
      discriminator(nf=2 * nf, name='discriminator_6'),
      discriminator(nf=2 * nf, name='discriminator_7'),
    ]

  def call(self, real_img=None, fake_img_list=None):
    real_x_list = []
    fake_x_list = []
    for i in range(len(fake_img_list)):
      real_x = tf.image.resize(real_img, fake_img_list[i].shape)
      real_x = self.discriminators[i](real_x)
      real_x_list.append(real_x)

      fake_x = self.discriminators[i](fake_img_list[i])
      fake_x_list.append(fake_x)
    return real_x_list, fake_x_list

def get_gan():
  Generator = generator_model()
  Discriminator = discriminator_model()
  gen_name = 'SinGAN'
  return Generator, Discriminator, gen_name


