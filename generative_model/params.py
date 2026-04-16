#设定默认参数
class Parameters():
    def __init__(self):
      self.epochs = 30000
      self.batch_size = 64
      self.lr_c = 0.0001
      self.lr_g = 0.0001
      self.crit_circle = 5
      self.generator_circle = 1
      self.z_dim = 29
      self.show_step = 100
      self.train_samples =2445