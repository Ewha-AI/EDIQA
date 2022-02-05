import torch
from torch import nn

import torchvision
from torchvision import models
import torchvision.transforms as transforms

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary

from PIL import Image
import numpy as np

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class AddNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(65)
        self.fn = fn
    def forward(self, x):
        return self.norm(self.fn(x) + x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 65),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(65, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, 65),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AddNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                AddNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*[self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool,
                                        self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4])

        self.Conv2dProjection = nn.Conv2d(2048, dim, (1, 1))
        
        self.maxpool = nn.MaxPool2d(2, 2)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.flatten = Rearrange('b c w h -> b c (w h)')

        self.pos_embedding = nn.Parameter(torch.randn(1, dim, int((image_size/(32*2))**2)+1))
        self.cls_token = nn.Parameter(torch.randn(1, dim, 1))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(mlp_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        # feature extraction
        x = self.backbone(img)

        # 2d projection
        x = self.Conv2dProjection(x)
        
        # max pooling
        x = self.maxpool(x)

        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape
        # print('parch embedding shape: ', x.shape)

        # flatten
        x = self.flatten(x)
        b, n, _ = x.shape

        # add class token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=-1)

        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = x[:, :, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)


if __name__ == '__main__':

    v = ViT(
        image_size = 512,
        patch_size = 32,
        num_classes=10,
        dim = 32,
        depth = 2, # encoder depth
        heads = 8,
        mlp_dim = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    
    img = Image.open('cat.jpg')

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    test = v(img)
    print(test)