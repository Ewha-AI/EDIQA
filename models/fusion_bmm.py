import torch
import torch.nn as nn

from .og_swin_backbone import MultiSwinTransformer as OGBackbone
from .fpn import FPN
from .attn import Attention
from .head import AvgPoolRegression
from .convnext import ConvNeXt
from .fpn import FPN
from .attn import Attention

class SwinConvBmm(nn.Module):
    def __init__(self):
        super().__init__()

        self.swin_transformer = OGBackbone()
        self.convnext = ConvNeXt() 
        self.attn = Attention(multi_feature=True, fpn=False)
        self.fpn = FPN(in_channels=[192, 384, 768, 768],
                       out_channels=256,
                       num_outs=4)
        self.head = AvgPoolRegression(fpn=True, feature_num=4)

        dims = [96, 192, 384, 768]

        self.downscale_layers = nn.ModuleList()
        for i in range(3):
            self.downscale_layers.append(nn.Conv2d(dims[i], dims[i] * 2, 2, 2))

    def forward(self, x):
        swin_features = list(self.swin_transformer(x))
        conv_features = list(self.convnext(x))

        for i in range(4):
            if i in range(3):
                # downscale convnext features
                conv_features[i] = self.downscale_layers[i](conv_features[i])
                     

        # attention
        swin_features = self.attn(swin_features)
        conv_features = self.attn(conv_features)

        # fpn network
        swin_features = list(self.fpn(swin_features))
        conv_features = list(self.fpn(conv_features))

        # bmm 
        features = []
        for i in range(4):
            b, c, w, h = conv_features[i].shape
            # print(conv_features[i].shape, swin_features[i].shape)
            conv_features[i] = conv_features[i].view(b, c, w * h).permute(0, 2, 1)
            swin_features[i] = swin_features[i].view(b, c, w * h)
            # print(conv_features[i].shape)
            # print(swin_features[i].shape)
            fusion = torch.bmm(swin_features[i], conv_features[i])
            features.append(fusion) 
            # print(fusion.shape)
        # exit()

        # head
        x = self.head(features)

        # print()
        # for i in range(4):
        #     print(x[i].shape)
        # exit()
        return x

        


