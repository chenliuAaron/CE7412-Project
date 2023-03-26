class Config(object):

    def __init__(self):
        # define some configures here
        self.x_dim = 1
        self.h_dim = 100
        self.z_dim = 16
        self.train_epoch = 5
        self.save_every = 1
        self.batch_size = 512
        self.device_ids = [0, 1]
        self.checkpoint_path = '../checkpoint/Epoch_6.pth'
        self.restore = True
