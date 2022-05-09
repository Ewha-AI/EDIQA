import torch
import torch.nn as nn
import torch.nn.functional as F


class SidePooling(nn.Module):
    def __init__(self):
        super().__init__()

    

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
        
        return tuple(outputs)