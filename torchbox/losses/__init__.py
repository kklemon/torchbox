from torchbox.losses.functional import standard_gan_loss_d, standard_gan_loss_g


class BaseGANLossFunction:
    name = None

    def loss_d(self, reals, fakes):
        raise NotImplementedError

    def loss_g(self, reals, fakes):
        raise NotImplementedError


class StandardGANLoss(BaseGANLossFunction):
    name = 'gan-loss'

    def __init__(self, D):
        self.D = D

    def loss_d(self, reals, fakes):
        return standard_gan_loss_d(self.D, reals, fakes)

    def loss_g(self, reals, fakes):
        return standard_gan_loss_g(self.D, fakes)

