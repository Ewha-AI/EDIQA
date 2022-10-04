import os, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaDIQaM_Model(nn.Module):
    """
    (Wa)DIQaM-NR Model
    """
    def __init__(self, weighted_average=True):
        """
        :param weighted_average: weighted average or not?
        """
        super(WaDIQaM_Model, self).__init__()
        self.conv1   = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3   = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4   = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5   = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6   = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7   = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8   = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9   = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10  = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1q_nr = nn.Linear(512, 512)
        self.fc2q_nr = nn.Linear(512, 1)
        self.fc1w_nr = nn.Linear(512, 512)
        self.fc2w_nr = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """
        # print(self.conv1(x).shape)
        # exit()
        h = F.relu(self.conv1(x))
        # print(h.shape)
        # exit()
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, 2)

        h = h.view(-1,512)

        return h

    def forward(self, x):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        batch_size = x[0].size(0) # x.size(0)
        n_patches = len(x) # x.size(1)

        if self.weighted_average:
            q = torch.ones((batch_size, 1)).cuda() #, device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)
        # print(q.size())

        for i in range(batch_size):

            h = self.extract_features(x[i])

            h_ = h  # save intermediate features

            # regression
            h = F.relu(self.fc1q_nr(h_))
            h = self.dropout(h)
            h = self.fc2q_nr(h)

            if self.weighted_average:
                w = F.relu(self.fc1w_nr(h_))
                w = self.dropout(w)
                w = F.relu(self.fc2w_nr(w)) + 0.000001  # small constant
                q[i] = torch.sum(h * w) / torch.sum(w)
            else:
                q[i * n_patches:(i + 1) * n_patches] = h

        return q