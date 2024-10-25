import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPoolRegression(nn.Module):
    def __init__(self, fpn=False, dim=256, feature_num=4):
        super(AvgPoolRegression, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if fpn and dim != 256:
            self.head = nn.Sequential(nn.Linear(dim*4, 1024),
                                      nn.GELU(),
                                      nn.Linear(1024, 1))
        elif fpn and feature_num == 4:
            self.head = nn.Sequential(nn.Linear(256*4, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 1))

        elif fpn and feature_num == 5:
            self.head = nn.Sequential(nn.Linear(256*5, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 1))

        else:
            # self.head = nn.Sequential(nn.Linear(2112, 1024),
            # self.head = nn.Sequential(nn.Linear(4224, 1024),
            self.head = nn.Sequential(nn.Linear(2880, 1024),
                                    nn.GELU(),
                                    nn.Linear(1024, 1))

    def forward(self, outs):

        xs = []
        for i in range(len(outs)):
            x = outs[i]
            if len(x.shape) == 4:
                b, c, h, w = x.shape
                # print(x.shape)
                x = x.view(b, c, h * w)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # print(x.shape)

            xs.append(x)

        x = torch.cat(xs, dim=1)
        x = self.head(x)

        return x