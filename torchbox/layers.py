import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.999, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Attention(nn.Module):
    def __init__(self, in_dim, key_query_dim, value_dim, n_heads=1, tau=1.0):
        super().__init__()

        self.query_w = nn.Linear(in_dim, key_query_dim)
        self.key_w = nn.Linear(in_dim, key_query_dim)
        self.value_w = nn.Linear(in_dim, value_dim)

        self.n_heads = n_heads
        self.kq_head_dim = key_query_dim // n_heads
        self.val_head_dim = value_dim // n_heads

        self.tau = tau

    def forward(self, query, key):
        bs, _, l = query.size()

        query_ = query.transpose(1, 2)
        key_ = key.transpose(1, 2)

        def reshape(x, head_dim):
            return x.view(bs, -1, self.n_heads, head_dim).transpose(1, 2)

        query = reshape(self.query_w(query_), self.kq_head_dim)
        key = reshape(self.key_w(key_), self.kq_head_dim).transpose(2, 3)
        value = reshape(self.value_w(key_), self.val_head_dim)

        attn = (query @ key) / math.sqrt(self.kq_head_dim)
        attn = attn / self.tau
        attn = F.softmax(attn, dim=-1)

        out = attn @ value
        out = out.transpose(1, 2).reshape(
            bs, l, self.n_heads * self.val_head_dim
        )
        out = out.permute(0, 2, 1)

        return out


class MinibatchStdDev(nn.Module):
    def __init__(self, alpha=1e-8):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        shape = x.shape
        batch_size = shape[0]

        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + self.alpha)
        y = y.mean().view([1] * len(shape))
        y = y.repeat(batch_size, 1, *shape[2:])
        y = torch.cat([x, y], 1)

        return y


class EqualizedConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *nn.modules.utils._single(kernel_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = np.prod(nn.modules.utils._single(kernel_size)) * c_in  # value of fan_in
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.conv1d(input=x,
                        weight=self.weight * self.scale,  # scale the weight on runtime
                        bias=self.bias if self.use_bias else None,
                        stride=self.stride,
                        padding=self.pad)


class EqualizedConvTranspose1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_in, c_out, *nn.modules.utils._single(kernel_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.conv_transpose1d(input=x,
                                  weight=self.weight * self.scale,  # scale the weight on runtime
                                  bias=self.bias if self.use_bias else None,
                                  stride=self.stride,
                                  padding=self.pad)



class EqualizedConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *nn.modules.utils._pair(kernel_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = np.prod(nn.modules.utils._pair(kernel_size)) * c_in  # value of fan_in
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.conv1d(input=x,
                        weight=self.weight * self.scale,  # scale the weight on runtime
                        bias=self.bias if self.use_bias else None,
                        stride=self.stride,
                        padding=self.pad)


class EqualizedConvTranspose2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_in, c_out, *nn.modules.utils._pair(kernel_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.conv_transpose2d(input=x,
                                  weight=self.weight * self.scale,  # scale the weight on runtime
                                  bias=self.bias if self.use_bias else None,
                                  stride=self.stride,
                                  padding=self.pad)


class PixelwiseNorm(nn.Module):
    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()
        y = x / y
        return y
