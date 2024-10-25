import torch
import torch.nn as nn

from .swin_transformer import SwinTransformer
from .fpn import FPN
from .attn import Attention
from .head import AvgPoolRegression
from .convnext import ConvNeXt


class EDIQA(nn.Module):
    def __init__(self, data_type):
        super().__init__()

        self.swin_transformer = SwinTransformer()
        self.convnext = ConvNeXt(data_type=data_type) 
        self.attn = Attention(multi_feature=True, fpn=True, dim=512)
        self.fpn = FPN(in_channels=[96, 192, 384, 768],
                       out_channels=256, 
                       num_outs=4)
        self.head = AvgPoolRegression(fpn=True, dim=512, feature_num=4) 

    def forward(self, x):
        features = []
        swin_features = list(self.swin_transformer(x))
        conv_features = list(self.convnext(x))

        # fpn network
        swin_features = self.fpn(swin_features)
        conv_features = self.fpn(conv_features)

        # concat features
        features = []
        for i in range(4):
            features.append(torch.cat([swin_features[i], conv_features[i]], axis=1))

        # attention
        x = self.attn(features)
#
        # head
        x = self.head(x)

        return x