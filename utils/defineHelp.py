import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class gradientLayer(nn.Module):
    '''
    	get the gradient of x, y direction
    '''
    def __init__(self):
        super(gradientLayer, self).__init__()
        self.weight_x = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
        		[[0,  -0.5, 0], [0,0,0], [0,0.5,0]]), 0), 0)
        self.weight_y = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
        		[[0,  0, 0], [-0.5,0,0.5], [0,0,0]]), 0), 0)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3,padding=1, bias=False)
        self.conv_x.weight.data = self.weight_x
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3,padding=1, bias=False)
        self.conv_y.weight.data = self.weight_y
    
    def forward(self, x):
        out_x = self.conv_x(x)
        out_y = self.conv_y(x)
        out = torch.cat((out_x, out_y), 1)
        return out

class shadingLayer(nn.Module):
    '''
    	get the shading layer based on input image and predicted reflectance
    '''
    def __init__(self):
        super(shadingLayer, self).__init__()
        self.eps = 1e-10
    def forward(self, img, ref):
        # img and reflectance are two NxCxHxW images
        if img.size()[0] != ref.size()[0] or \
        	img.size()[2] != ref.size()[2] or\
        	img.size()[3] != ref.size()[3]:
        		print img.size()
        		print ref.size()
        		raise ValueError('size of image and reflectance does not match')
        
        meanImg = torch.mean(img, dim=1)
        meanImg = torch.unsqueeze(meanImg, dim=1)
        output = meanImg/(ref + self.eps)
        return output

class gradientLayer_color(nn.Module):
    '''
    	get the gradient of x, y direction
    '''
    def __init__(self):
        super(gradientLayer_color, self).__init__()
        self.weight_x = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
        		[[0,  -0.5, 0], [0,0,0], [0,0.5,0]]), 0), 0)
        self.weight_y = torch.unsqueeze(torch.unsqueeze(torch.Tensor(
        		[[0,  0, 0], [-0.5,0,0.5], [0,0,0]]), 0), 0)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3,padding=1, bias=False)
        self.conv_x.weight.data = self.weight_x
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3,padding=1, bias=False)
        self.conv_y.weight.data = self.weight_y
    
    def forward(self, x):
        # number of channel
        # channel 1
        tmpX_0 = x[:,0,:,:].unsqueeze(dim=1)
        outx_0 = self.conv_x(tmpX_0)
        outy_0 = self.conv_y(tmpX_0)
        
        tmpX_1= x[:,1,:,:].unsqueeze(dim=1)
        outx_1 = self.conv_x(tmpX_1)
        outy_1 = self.conv_y(tmpX_1)
        
        tmpX_2= x[:,2,:,:].unsqueeze(dim=1)
        outx_2 = self.conv_x(tmpX_2)
        outy_2 = self.conv_y(tmpX_2)
        
        out = torch.cat((outx_0, outx_1, outx_2, outy_0, outy_1, outy_2), 1)
        return out

class shadingLayer_color(nn.Module):
    '''
    	get the shading layer based on input image and predicted reflectance
    '''
    def __init__(self):
        super(shadingLayer_color, self).__init__()
        self.eps = 1e-10
    def forward(self, img, ref):
        # img and reflectance are two NxCxHxW images
        if img.size()[0] != ref.size()[0] or \
        	img.size()[2] != ref.size()[2] or\
        	img.size()[3] != ref.size()[3]:
        		print img.size()
        		print ref.size()
        		raise ValueError('size of image and reflectance does not match')
        
        output = img/(ref + self.eps)
        return output
