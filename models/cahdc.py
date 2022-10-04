import os, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SiPNet(nn.Module):
    def __init__(self,num):
        super(SiPNet,self).__init__()
        if num == 1:
            self.features=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,padding=0,stride=1),
            nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2),
            nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2),
            nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2)
        )
        elif num == 2:
            self.features = nn.Sequential(
                nn.Conv2d(64,64,kernel_size=1,padding=0,stride=1),
                nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2),
                nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2)
            ) 
        elif num ==3:
            self.features = nn.Sequential(
                nn.Conv2d(64,64,kernel_size=1,padding=0,stride=1),
                nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2),

            )
        else :
            self.features = nn.Conv2d(512,64,kernel_size=1,padding=0,stride=1)
        
        


        #weight_init(self.features)
    
    def forward(self,x):
        x = self.features(x)
        return x



        

class caHDCModel(nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        super(caHDCModel, self).__init__()
        self.nc= 3 #opt.n_channels
        self.nun_classes=1
        self.conv1=nn.Sequential(
            nn.Conv2d(self.nc,32,kernel_size=3,padding=1,stride=1),
            nn.Conv2d(32,32,kernel_size=3,padding=1,stride=1),
            
        )
        weight_init(self.conv1)
        self.maxp1=nn.MaxPool2d(kernel_size=2,padding=1,stride=2)
        self.conv2=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=1,stride=1),
            nn.Conv2d(32,32,kernel_size=3,padding=1,stride=1)
        )
        weight_init(self.conv2)
        
        self.maxp2=nn.MaxPool2d(kernel_size=2,padding=1,stride=2)
        
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
            nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1),
            nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)
        )
        weight_init(self.conv3)
        self.spin1= SiPNet(num=1)
        weight_init(self.spin1)
        self.maxp3=nn.MaxPool2d(kernel_size=2,padding=1,stride=2)
        
        self.conv4=nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)
        weight_init(self.conv4)
        self.spin2=SiPNet(num=2)
        weight_init(self.spin2)
        self.maxp4=nn.MaxPool2d(kernel_size=2,padding=1,stride=2)
        
        self.conv5=nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)
        weight_init(self.conv5)
        self.spin3=SiPNet(num=3)
        weight_init(self.spin3)
        self.maxp5=nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
        
        self.conv6=nn.Conv2d(64,512,kernel_size=3,padding=1,stride=1)
        weight_init(self.conv6)
        self.spin4=SiPNet(num=4)
        weight_init(self.spin4)
        self.maxp6=nn.MaxPool2d(kernel_size=10,stride=10)

        # self.fc= nn.Linear(256,self.nun_classes)
        self.fc1= nn.Conv2d(256, 100, kernel_size=1)
        self.fc2= nn.Conv2d(100, 1, kernel_size=1)
        weight_init(self.fc1)
        weight_init(self.fc2)


    def forward(self,x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x1 = self.spin1(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        x2 = self.spin2(x)
        x = self.maxp4(x)
        x = self.conv5(x)
        x3 = self.spin3(x)
        x = self.maxp5(x)
        x = self.conv6(x)
        x4 = self.spin4(x)
        
        x1 = self.maxp6(x1)
        x2 = self.maxp6(x2)
        x3 = self.maxp6(x3)
        x4 = self.maxp6(x4)

        x = torch.cat((x1,x2,x3,x4),dim=1)
        # x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x=self.fc2(x)
        # print(x.shape)
        x = x.squeeze(1)
        x = x.squeeze(1)
        x = x.squeeze(1)
        # print(x.shape)
        # exit()
        x = rearrange(x, 'b -> b 1')
        return x


def weight_init(net): 
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
