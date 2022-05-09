import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Attention model for implementing channel and spatial attention in IQA
    :param input_feature: feature maps from FPN or backbone
    :param n_quality_levels: 1 for MOS prediction and 5 for score distribution
    :param name: name of individual layers
    :param return_feature_map: flag to return feature map or not
    :param return_features: flag to return feature vector or not
    :return: output of attention module
    """
    def __init__(self, input_feature=None, name=None, return_feature_map=False, padding=False, n_quality_levels=1, multi_feature=True, fpn=False, dim=256, return_features=False):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3,bias=False),
                                        nn.Sigmoid())

        # feature_width = [126, 62, 30, 14, 7]
        # feature_width = [16, 16, 16, 16]
        # feature_width = [28, 14, 7, 7]
        if dim != 256:
            dim = [dim] * 4
        elif fpn:
            dim = [256]* 4
        elif padding:
            dim = [96, 192, 384, 768]
        else:
            dim = [192, 384, 768, 768]
        self.shared_dense_layer = nn.ModuleList()

        if multi_feature == False:
            self.shared_dense_layer.append(nn.Sequential(nn.Linear(768, 768),
                                                         nn.Sigmoid()))
        else:
            for i in range(4):
                # self.shared_dense_layer.append(nn.Sequential(nn.Linear(768, 768), # 256
                #                             nn.Sigmoid()))
                self.shared_dense_layer.append(nn.Sequential(nn.Linear(dim[i], dim[i]), # 256
                                            nn.Sigmoid()))

    def forward(self, x):

        outputs = []
        for i in range(len(x)):
            b, l, h, w = x[i].size() # [32, 1536, 7, 7]

            avg_pool_channel = self.gap(x[i])
            avg_pool_channel = torch.flatten(avg_pool_channel, 1)
            avg_pool_channel =  self.shared_dense_layer[i](avg_pool_channel)
            max_pool_channel = self.gmp(x[i])
            max_pool_channel = torch.flatten(max_pool_channel, 1)
            max_pool_channel =  self.shared_dense_layer[i](max_pool_channel)
            channel_weights = (avg_pool_channel + max_pool_channel) / 2 # [32, 1536]
            channel_weights = channel_weights.view(b, l, 1)
            channel_weights = torch.cat([channel_weights,] * (h * w), dim=-1)
            channel_weights = channel_weights.view(b, l, h, w)

            avg_pool_spatial = torch.mean(x[i], 1)
            avg_pool_spatial = avg_pool_spatial.view(b, 1, h, w)
            max_pool_spatial, _ = torch.max(x[i], 1)
            max_pool_spatial = max_pool_spatial.view(b, 1, h, w)
            spatial_weights = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1) # 32, 2, 7, 7
            spatial_weights = self.conv_layer(spatial_weights)
            spatial_weights = torch.cat([spatial_weights,] * l, dim=1)

            out = torch.mul(torch.mul(channel_weights, x[i]), spatial_weights)
            outputs.append(out)
        #     print(out.shape)
        # exit()
        
        return tuple(outputs)