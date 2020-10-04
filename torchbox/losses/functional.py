import torch
import torch.nn.functional as F


def standard_gan_loss_d(D, reals, fakes):
    real_scores = D(reals)
    real_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))

    fake_scores = D(fakes)
    fake_loss = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))

    loss = (real_loss + fake_loss) / 2.0

    return loss


def standard_gan_loss_g(D, fakes):
    fakes_score = D(fakes)
    fakes_loss = F.binary_cross_entropy_with_logits(fakes_score, torch.ones_like(fakes_score))

    return fakes_loss


def gradient_penalty(D, reals, fakes, reg_lambda=10):
    is_sample_list = True
    if not isinstance(reals, (list, tuple)):
        assert not isinstance(fakes, (list, tuple))
        reals = [reals]
        fakes = [fakes]
        is_sample_list = False

    reals = [real.detach() for real in reals]
    fakes = [fake.detach() for fake in fakes]

    batch_size = reals[0].size(0)
    num_dims = reals[0].ndim - 1

    # Draw random epsilon
    eps = torch.rand((batch_size, *[1] * num_dims)).to(reals[0].device)

    # Create the merge of both real and fake samples
    merged = [eps * real + (1 - eps) * fake for real, fake in zip(reals, fakes)]
    for sample in merged:
        sample.requires_grad = True

    op = D(merged if is_sample_list else merged[0])

    # Perform backward pass from op to merged for obtaining the gradients
    gradients = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)

    # Calculate the penalty using these gradients
    penalties = [(gradient.reshape(gradient.size(0), -1).norm(p=2, dim=1) ** 2).mean().unsqueeze(0)
                 for gradient in gradients]
    penalty = reg_lambda * torch.cat(penalties).mean()

    return penalty


def wasserstein_gan_loss_d(D, reals, fakes, drift=0.001):
    fake_scores = D(fakes)
    real_scores = D(reals)

    loss = fake_scores.mean() - real_scores.mean() + drift * (real_scores ** 2).mean()

    return loss


def wasserstein_gan_loss_g(D, fakes):
    return -D(fakes).mean()


def relativistic_average_gan_loss_d(D, reals, fakes, f1, f2):
    real_out = D(reals)
    fake_out = D(fakes)

    rf_diff = real_out - fake_out.mean()
    fr_diff = fake_out - real_out.mean()

    loss = f1(rf_diff).mean() + f2(fr_diff).mean()

    return loss / 2.0


def relativistic_average_gan_loss_g(D, reals, fakes, g1, g2):
    return relativistic_average_gan_loss_d(D, fakes, reals, g1, g2)
