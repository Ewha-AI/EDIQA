import torch
import torch.nn as nn

from .og_swin_backbone import MultiSwinTransformer as OGBackbone
from .fpn import FPN
from .attn import Attention
from .head import AvgPoolRegression
from .convnext import ConvNeXt
from .fpn import FPN
from .attn import Attention

class SwinConvConcat(nn.Module):
    def __init__(self):
        super().__init__()

        self.swin_transformer = OGBackbone()
        self.convnext = ConvNeXt() 
        self.attn = Attention(multi_feature=True, fpn=True, dim=256)
        self.fpn = FPN(in_channels=[192, 384, 768, 768],
                       out_channels=128, # 256
                       num_outs=4)
        self.head = AvgPoolRegression(fpn=True, dim=256, feature_num=4) # 512

        dims = [96, 192, 384, 768]

        self.downscale_layers = nn.ModuleList()
        for i in range(3):
            self.downscale_layers.append(nn.Conv2d(dims[i], dims[i] * 2, 2, 2))

    def forward(self, x):
        features = []
        swin_features = list(self.swin_transformer(x))
        conv_features = list(self.convnext(x))

        for i in range(4):
            if i in range(3):
                # downscale convnext features
                conv_features[i] = self.downscale_layers[i](conv_features[i])
            
            # # bmm 
            # b, c, w, h = conv_features[i].shape
            # # print(conv_features[i].shape, swin_features[i].shape)
            # conv_features[i] = conv_features[i].view(b, c, w * h).permute(0, 2, 1)
            # swin_features[i] = swin_features[i].view(b, c, w * h)
            # # print(conv_features[i].shape)
            # # print(swin_features[i].shape)
            # fusion = torch.bmm(swin_features[i], conv_features[i])
            # features.append(fusion)          

        # return tuple(features)

        # fpn network
        swin_features = self.fpn(swin_features)
        conv_features = self.fpn(conv_features)

        # concat features
        features = []
        for i in range(4):
            features.append(torch.cat([swin_features[i], conv_features[i]], axis=1))
            # print(features[i].shape)

        # attention
        x = self.attn(features)

        # head
        x = self.head(x)

        # print()
        # for i in range(4):
        #     print(x[i].shape)
        # exit()
        return x

        


