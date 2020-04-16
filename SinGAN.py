from SinGAN_Block import *

class generator_model(tf.keras.Model):
  def __init__(self):
    super(generator_model, self).__init__()
    self.nf = 32
    self.stages = 1
    self.generator_first = generator_First(self.nf, name='generator_0')
    self.generator_middle = []
  def prepare_model(self, z_list):
    nf = self.nf
    for i in range(len(z_list)):
      self.stages += 1
      if (i + 2) % 4 == 0:
        self.generator_middle.append(generator_Middle(nf, name='generator_{}'.format(i+1)))
      else:
        nf = nf * 2
        self.generator_middle.append(generator_Middle(nf, name='generator_{}'.format(i+1)))
  def call(self, image, z_list=None):
    x_list = []
    for i in range(len(z_list)):
      if i == 0:
        x = self.generator_first(z_list[0])
        x_list.append(x)
      else:

        x = tf.image.resize(x_list[i-1], (z_list[i].shape[1], z_list[i].shape[2]))
        x = self.generator_middle[i-1](x, z_list[i])
        x_list.append(x)
    return x_list

class discriminator_model(tf.keras.Model):
  def __init__(self):
    super(discriminator_model, self).__init__()
    self.stages = 0
    self.nf = 32
    self.discriminators = []
  def prepare_model(self, z_list):
    nf = self.nf
    for i in range(len(z_list)):
      self.stages += 1
      if (i + 1) % 4 == 0:
        self.discriminators.append(discriminator_block(nf, name='discriminator_{}'.format(i)))
      else:
        nf = nf * 2
        self.discriminators.append(discriminator_block(nf, name='discriminator_{}'.format(i)))

  def call(self, img=None, stage=None):
    x = self.discriminators[stage](img)
    return x

def get_gan():
  Generator = generator_model()
  Discriminator = discriminator_model()
  gen_name = 'SinGAN'
  return Generator, Discriminator, gen_name


