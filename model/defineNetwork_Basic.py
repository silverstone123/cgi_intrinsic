'''
    define basic network structures
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
#sys.path.append('/vulcan/scratch/hzhou/code/project_cvpr2019/code/functional-zoo/')
#from visualize import make_dot
import time

# define 3x3 convolution 
def conv3X3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, bias=False)

# define residual block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, padding=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = conv3X3(outplanes, outplanes, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(outplanes)
        
        self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.inplanes != self.outplanes:
        		out += self.shortcuts(x)
        else:
        		out += x
        
        out = F.relu(out)
        return out
