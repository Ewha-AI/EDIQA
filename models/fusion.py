import torch
import torch.nn as nn


class Fusionmodule(nn.Module):
    def __init__(self,out_channels):
        super(Fusionmodule,self).__init__()
        
        self.selected = nn.Sequential(
               nn.AdaptiveAvgPool2d(output_size = 1),
               nn.Conv2d(out_channels, 4, kernel_size=1, stride=1,bias=True),
               nn.ReLU(inplace=True)
                )
        self.bot = nn.Sequential(
               nn.AdaptiveMaxPool2d(output_size = 1),
               nn.Conv2d(out_channels, 4, kernel_size=1, stride=1,bias=True),
               nn.ReLU(inplace=True)
                )
        self.fcs = nn.ModuleList([])
        for i in range(4):
            self.fcs.append(nn.Conv2d(4, out_channels, kernel_size=1, stride=1,bias=True))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,inp_feats):
        
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        print(inp_feats[0].shape)
        print(n_feats)
        inp_feats = torch.cat(inp_feats, dim=1)
        print(inp_features.shape)
        inp_feats = inp_feats.view(batch_size, 4, n_feats,inp_feats.shape[2], inp_feats.shape[3])
        print(inp_features.shape)
        exit()
        feature_sum = torch.sum(inp_feats,dim=1)
        selected_feat = self.selected(feature_sum)
        selected_bot = self.bot(feature_sum)
        selected_final = selected_feat+selected_bot
        atten_vectors = [fc(selected_final) for fc in self.fcs]
        atten_vectors = torch.cat(atten_vectors,dim=1)
        atten_vectors = atten_vectors.view(batch_size, 4, n_feats,1,1)#inp_feats[0].shape[2], inp_feats[0].shape[3])
        atten_vectors = self.softmax(atten_vectors)
        
        out = torch.sum(inp_feats*atten_vectors,dim=1)
        return out