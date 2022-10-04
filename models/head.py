import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, in_channels=256, representation_size=256):
        super(Head, self).__init__()

        # self.hidden_dim = 512
        # feature_width = [56, 28, 14, 7, 4]
        feature_width = [126, 62, 30, 14, 7]

        self.fc1 = nn.Linear(in_channels * (feature_width[0] ** 2), 1)
        self.fc2 = nn.Linear(in_channels * (feature_width[1] ** 2), 1)
        self.fc3 = nn.Linear(in_channels * (feature_width[2] ** 2), 1)
        self.fc4 = nn.Linear(in_channels * (feature_width[3] ** 2), 1)
        self.fc5 = nn.Linear(in_channels * (feature_width[4] ** 2), 1)

    def forward(self, outs):

        x1 = outs[0].flatten(start_dim=1)
        x2 = outs[1].flatten(start_dim=1)
        x3 = outs[2].flatten(start_dim=1)
        x4 = outs[3].flatten(start_dim=1)
        x5 = outs[4].flatten(start_dim=1)

        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))
        x3 = F.relu(self.fc3(x3))
        x4 = F.relu(self.fc4(x4))
        x5 = F.relu(self.fc5(x5))

        # outs = [x1, x2, x3, x4, x5]
        # outs = [x1]
        outs = x1 + x2 + x3 + x4
        outs = outs / 4

        # return tuple(outs)
        return outs


class Head2(nn.Module):
    def __init__(self, in_channels=768, representation_size=256): # 256
        super(Head2, self).__init__()

        # self.hidden_dim = 512
        # feature_width = [56, 28, 14, 7, 4]
        # feature_width = [126, 62, 30, 14, 7]
        feature_width = [7] * 5

        self.fc1 = nn.Sequential(nn.Linear(in_channels * (feature_width[0] ** 2), representation_size * 4),
                                 nn.GELU(),
                                 nn.Linear(representation_size * 4, representation_size))
        self.fc2 = nn.Sequential(nn.Linear(in_channels * (feature_width[1] ** 2), representation_size * 4),
                                 nn.GELU(),
                                 nn.Linear(representation_size * 4, representation_size))
        self.fc3 = nn.Sequential(nn.Linear(in_channels * (feature_width[2] ** 2), representation_size * 4),
                                 nn.GELU(),
                                 nn.Linear(representation_size * 4, representation_size))
        self.fc4 = nn.Sequential(nn.Linear(in_channels * (feature_width[3] ** 2), representation_size * 4),
                                 nn.GELU(),
                                 nn.Linear(representation_size * 4, representation_size))
        # self.fc5 = nn.Sequential(nn.Linear(in_channels * (feature_width[4] ** 2), representation_size * 4),
        #                          nn.ReLU(),
        #                          nn.Linear(representation_size * 4, representation_size))

        self.head = nn.Linear(representation_size * 4, 1)

    def forward(self, outs):

        x1 = outs[0].flatten(start_dim=1)
        x2 = outs[1].flatten(start_dim=1)
        x3 = outs[2].flatten(start_dim=1)
        x4 = outs[3].flatten(start_dim=1)
        # x5 = outs[4].flatten(start_dim=1)

        x1 = F.gelu(self.fc1(x1))
        x2 = F.gelu(self.fc2(x2))
        x3 = F.gelu(self.fc3(x3))
        x4 = F.gelu(self.fc4(x4))
        # x5 = F.relu(self.fc5(x5))

        x = torch.cat([x1, x3, x3, x4], axis=1)

        x = self.head(x)

        # return tuple(outs)
        return x


class Head3(nn.Module):
    def __init__(self, in_channels=768, representation_size=256):
        super(Head3, self).__init__()

        # self.hidden_dim = 512
        # feature_width = [56, 28, 14, 7, 4]
        feature_width = [16]

        self.fc1 = nn.Linear(in_channels * (feature_width[0] ** 2), 1)

    def forward(self, outs):

        # print(len(outs))
        # print(outs[0].shape)
        outs = outs[0]
        x1 = outs.flatten(start_dim=1)
        # print(x1.shape)
        # exit()

        x1 = F.relu(self.fc1(x1))

        # outs = [x1, x2, x3, x4, x5]
        # outs = [x1]
        outs = x1

        # return tuple(outs)
        return outs

class Neck(nn.Module):
    def __init__(self, in_channels=768, representation_size=256):
        super(Neck, self).__init__()

    def forward(self, outs):

        x = torch.cat([outs[i] for i in range(len(outs))], axis=1)

        return x

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
            # print(x.shape)
            # print(len(x.shape))
            # exit()
            if len(x.shape) == 4:
                b, c, h, w = x.shape
                # print(x.shape)
                x = x.view(b, c, h * w)
            # print(x.shape)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # print(x.shape)
            # exit()
            xs.append(x)

        x = torch.cat(xs, dim=1)
        # print(x.shape)
        # exit()
        x = self.head(x)

        return x
            


