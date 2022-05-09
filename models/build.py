from .multi_swin_transformer import MultiSwinTransformer
from .multi_pswin_transformer import MultiPSwinTransformer
from .backbone_mpswin_transformer import MultiPSwinTransformer as SwinBackbone
from .SwinT_modified2 import SwinTransformer as SwinTModified
from .fpn import FPN
from .head import Head, Head2, Head3, Neck, AvgPoolRegression
from .attn import Attention
from .fusion import Fusionmodule as Fusion
from .mixer import MLPMixer
from .token_swin_transformer import MultiSwinTransformer as TokenSwin
from .og_swin_backbone import MultiSwinTransformer as OGBackbone
from .og_pswin_backbone import MultiSwinTransformer as OGPBackbone
from .convnext import ConvNeXt
from .fusion_backbone import SwinConvConcat
from .fusion_bmm import SwinConvBmm

import torch
import torch.nn as nn



def build_model(config):
    model_type = config
    
    if model_type == 'multi_swin':
        model = MultiSwinTransformer()

    elif model_type == 'multi_pswin':
        model = MultiPSwinTransformer()

    elif model_type == 'multi_swin512_fpn':
        model = nn.Sequential(SwinBackbone(),
                              FPN(in_channels=[96, 192, 384, 768],
                                  out_channels=256,
                                  num_outs=5),
                              Head())
                            
    elif model_type == 'multi_swin512_fpn_attn':
        model = nn.Sequential(SwinBackbone(),
                              FPN(in_channels=[96, 192, 384, 768],
                                  out_channels=256,
                                  num_outs=5),
                              Attention(),
                              Head())

    elif model_type == 'multi_swin512_fpn_attn_head2':
        model = nn.Sequential(SwinBackbone(),
                              FPN(in_channels=[96, 192, 384, 768],
                                  out_channels=256,
                                  num_outs=5),
                              Attention(),
                            #   Fusion(256),
                              Head2())

    elif model_type == 'multi_swin512_768x16x16_attn_head2':
        model = nn.Sequential(SwinBackbone(),
                              Attention(multi_feature=True),
                              Head2())

    elif model_type == 'multi_swin512_768x16x16_attn_mixer':
        model = nn.Sequential(SwinBackbone(),
                              Attention(multi_feature=True),
                              Neck(),
                              MLPMixer())

    # elif model_type == 'multi_swin224_attn_head2':
    #     model = nn.Sequential()

    elif model_type == 'swin_avgpool_attn_reg':
        model = nn.Sequential(OGBackbone(),
                              Attention(multi_feature=True),
                              AvgPoolRegression()
                              )

    elif model_type == 'swin_avgpool_fpn_attn_reg':
        model = nn.Sequential(OGBackbone(),
                              FPN(in_channels=[192, 384, 768, 768],
                                  out_channels=256,
                                  num_outs=4),
                              Attention(multi_feature=True, fpn=True),
                              AvgPoolRegression(fpn=True)
                              )

    elif model_type == 'swin_avgpool_fpn_reg':
        model = nn.Sequential(OGBackbone(),
                              FPN(in_channels=[192, 384, 768, 768],
                                  out_channels=256,
                                  num_outs=4),
                              AvgPoolRegression(fpn=True)
                              )

    elif model_type == 'swin_avgpool_attn_fpn_reg':
        model = nn.Sequential(OGBackbone(),
                              Attention(multi_feature=True, fpn=False),
                              FPN(in_channels=[192, 384, 768, 768],
                                  out_channels=256,
                                  num_outs=4),
                              AvgPoolRegression(fpn=True, feature_num=4)
                              )

    elif model_type == 'swin_reg':
        model = nn.Sequential(OGBackbone(),
                              AvgPoolRegression()
                              )

    elif model_type == 'pswin_attn_fpn_reg':
        model = nn.Sequential(SwinBackbone(),
                              Attention(multi_feature=True, padding=True, fpn=False),
                              FPN(in_channels=[96, 192, 384, 768],
                                  out_channels=256,
                                  num_outs=4),
                              AvgPoolRegression(fpn=True, feature_num=4))

    elif model_type == 'swin_conv_concat':
        model = nn.Sequential(SwinConvConcat())

    elif model_type == 'swin_conv_bmm':
        model = nn.Sequential(SwinConvBmm())


    elif model_type == 'og':
        model = SwinTModified()

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
    