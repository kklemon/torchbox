import torch
import torch.nn.functional as F


def standard_gan_loss_d(D, reals, fakes):
    reals_score = D(reals)
    reals_loss = F.binary_cross_entropy_with_logits(reals_score, torch.ones_like(reals_score))

    fakes_score = D(fakes)
    fakes_loss = F.binary_cross_entropy_with_logits(fakes_score, torch.zeros_like(fakes_score))

    loss = (reals_loss + fakes_loss) / 2.0

    return loss


def standard_gan_loss_g(D, fakes):
    fakes_score = D(fakes)
    fakes_loss = F.binary_cross_entropy_with_logits(fakes_score, torch.ones_like(fakes_score))

    return fakes_loss
