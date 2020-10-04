import torch.nn.functional as F

from torchbox.losses.functional import (
    standard_gan_loss_d,
    standard_gan_loss_g,
    gradient_penalty,
    wasserstein_gan_loss_d,
    wasserstein_gan_loss_g,
    relativistic_average_gan_loss_d,
    relativistic_average_gan_loss_g
)


class BaseGANLossFunction:
    def loss_d(self, reals, fakes):
        raise NotImplementedError

    def loss_g(self, reals, fakes):
        raise NotImplementedError


class StandardGANLoss(BaseGANLossFunction):
    def __init__(self, D):
        self.D = D

    def loss_d(self, reals, fakes):
        return standard_gan_loss_d(self.D, reals, fakes)

    def loss_g(self, reals, fakes):
        return standard_gan_loss_g(self.D, fakes)


class WassersteinGANLoss(BaseGANLossFunction):
    def __init__(self, D, use_gp=True, reg_lambda=10, drift=0.001, gp_every=None):
        self.D = D
        self.use_gp = use_gp
        self.reg_lambda = reg_lambda
        self.drift = drift
        self.gp_every = gp_every
        self.step = 0

    def gradient_penalty(self, reals, fakes):
        return gradient_penalty(self.D, reals, fakes, reg_lambda=self.reg_lambda)

    def loss_d(self, reals, fakes, update_step=True):
        loss = wasserstein_gan_loss_d(self.D, reals, fakes, drift=self.drift)

        if self.use_gp and self.reg_lambda and (not self.gp_every or self.step % self.gp_every == 0):
            loss += self.gradient_penalty(reals, fakes)

        if update_step:
            self.step += 1

        return loss

    def loss_g(self, reals, fakes):
        return wasserstein_gan_loss_g(self.D, fakes)


class RelativisticAverageGANLoss(BaseGANLossFunction):
    def __init__(self, D, f1, f2, g1, g2):
        self.D = D
        self.f1 = f1
        self.f2 = f2
        self.g1 = g1
        self.g2 = g2

    def loss_d(self, reals, fakes):
        return relativistic_average_gan_loss_d(self.D, reals, fakes, self.f1, self.f2)

    def loss_g(self, reals, fakes):
        return relativistic_average_gan_loss_g(self.D, reals, fakes, self.g1, self.g2)


class RelativisticAverageHingeGANLoss(RelativisticAverageGANLoss):
    def __init__(self, D):
        f1 = lambda x: F.relu(1 - x)
        f2 = lambda x: F.relu(1 + x)
        super().__init__(D, f1, f2, f1, f2)
