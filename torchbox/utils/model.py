import warnings
import torch.nn as nn


def update_average(model_tgt, model_src, beta):
    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def update(self, model):
        self.model = model


class AveragedModelWrapper(ModelWrapper):
    def __init__(self, model, beta=0.999):
        super().__init__(model)
        self.beta = beta

    def update(self, model):
        if self.model is model:
            warnings.warn(f'Given mode is same as wrapped one. You probably were intending to pass of copy  of that '
                          f'mode to the constructor.')
        update_average(self.model, model, self.beta)
