import os
import numpy as np
import torch
from torch import nn
import random
import scipy
import scipy.stats
from scipy.stats import pearsonr
import math
from torch.optim.lr_scheduler import _LRScheduler
from math import log2

from copy import deepcopy
from collections import OrderedDict
from sys import stderr
from torch import Tensor


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_distribution(score_scale, mean, std, distribution_type='standard'):
    """
    Calculate the distribution of scores from MOS and standard distribution, two types of distribution are supported:
        standard Gaussian and Truncated Gaussian
    :param score_scale: MOS scale, e.g., [1, 2, 3, 4, 5]
    :param mean: MOS
    :param std: standard deviation
    :param distribution_type: distribution type (standard or truncated)
    :return: Distribution of scores
    """
    if distribution_type == 'standard':
        # import matplotlib.pyplot as plt
        distribution = scipy.stats.norm(loc=mean, scale=std)
        # x = np.arange(0, 11, 0.01) #X 확률변수 범위
        # y = distribution.pdf(x) #X 범위에 따른 정규확률밀도값
        # fig, ax = plt.subplots(1,1)
        # ax.plot(x, y,'bo', ms=8, label = 'normal pdf')
        # ax.vlines(x, 0, y, colors='b', lw =5, alpha =0.5) #결과는
        # ax.set_ylim([0,5]) #y축 범위
        # plt.savefig('norm_test.png')
        # print(mean, std)
    else:
        distribution = scipy.stats.truncnorm((score_scale[0] - mean) / std, (score_scale[-1] - mean) / std, loc=mean, scale=std)
    score_distribution = []
    for s in score_scale:
        score_distribution.append(distribution.pdf(s))

    return score_distribution

def distribution_to_score(score_scale, distribution):
    b, _ = distribution.shape
    score = np.multiply([score_scale]*b, distribution.detach().cpu().squeeze().numpy())
    score = np.sum(score, axis=1)

    return score

def calculate_plcc(pred, gt):
    """
    pred: prediction mos score by model
    gt: ground truth mos score
    """
    pred = pred.detach().cpu().squeeze()
    gt = gt.detach().cpu().squeeze()
    cval, _ = pearsonr(pred, gt)

    return cval * 100

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



# calculate the kl divergence
def kl_divergence(p, q):
	return sum([p[i] * log2(p[i]/q[i]) for i in range(len(p))])


class EMA(nn.Module):
    # https://www.zijianhu.com/post/pytorch/ema/
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: Tensor, return_feature: bool = False) -> Tensor:
        if self.training:
            return self.model(inputs, return_feature)
        else:
            return self.shadow(inputs, return_feature)