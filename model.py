import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class VGGnet(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.layer_10 = self.conv2d_layer(3,64)  
        self.layer_11=self.conv2d_layer(64,64)

        self.layer_20=self.conv2d_layer(64,128)
        self.layer_21=self.conv2d_layer(128,128)

        self.layer_30=self.conv2d_layer(128,256)
        self.layer_31=self.conv2d_layer(256,256)
        self.layer_32=self.conv2d_layer(256,256)
        
        self.layer_40=self.conv2d_layer(256,512)
        self.layer_41=self.conv2d_layer(512,512)
        self.layer_42=self.conv2d_layer(512,512)

        

        self.downsample= nn.MaxPool2d(2, stride=2, return_indices=True) 
        
        self.linear_b_0=nn.Linear(73728,50)
        self.linear_b_1=nn.Linear(50,1)

        self.flat=nn.Flatten()
        
    def conv2d_layer(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1):
        """A function creating a layer consisting of convolution, batch normalisation and relu.
        @params
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            padding (int, optional): Padding for the convolution stage. Default is 1.
        
        Returns:
            A layer made up of three smaller layers.
        
        """

        layer=[]
        layer.append(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding,stride=stride))
        layer.append(nn.BatchNorm2d(out_ch))
        layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)
    
 
    def forward(self,x):

        x=self.layer_10(x)
        x=self.layer_11(x)
        x,i1=self.downsample(x)
      
        x=self.layer_20(x)
        x=self.layer_21(x)
        x,i2=self.downsample(x)
 

        x=self.layer_30(x)
        x=self.layer_31(x)
        x=self.layer_32(x)
        x,i3=self.downsample(x)
        
        x=self.layer_40(x)
        x=self.layer_41(x)
        x=self.layer_42(x)
        x,i4=self.downsample(x)
    

        flat=self.flat(x)

        
        b_0=(F.relu(self.linear_b_0(flat)))
        b_=F.relu(self.linear_b_1(b_0))
        
        return b_
