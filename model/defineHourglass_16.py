import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append('/scratch1/intrinsicImage/myProject/code_synthetic/functional-zoo/')
from visualize import make_dot
import time

# we define Hour Glass network based on the paper
# Stacked Hourglass Networks for Human Pose Estimation
#       Alejandro Newell, Kaiyu Yang, and Jia Deng
# the code is adapted from
# https://github.com/umich-vl/pose-hg-train/blob/master/src/models/hg.lua

# we suppose the input images is with size 16x16, we can use something like a convolution 
# to deal with full-size image



def conv3X3(in_planes, out_planes, stride=1):
		"""3x3 convolution with padding"""
		return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
								padding=1, bias=False)
# define the network
class BasicBlock(nn.Module):
		def __init__(self, inplanes, outplanes, stride=1, downsample=None):
				super(BasicBlock, self).__init__()
				self.inplanes = inplanes
				self.outplanes = outplanes
				self.conv1 = conv3X3(inplanes, outplanes, 1)
				self.bn1 = nn.BatchNorm2d(outplanes)
				self.conv2 = conv3X3(outplanes, outplanes, 1)
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

class HourglassBlock(nn.Module):
		def __init__(self, inplane, middleNet):
				super(HourglassBlock, self).__init__()
				# upper branch
				self.upper = BasicBlock(inplane, inplane)

				# lower branch
				self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
				self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
				self.low1 = BasicBlock(inplane, 2*inplane)
				self.low2 = middleNet
				self.low3 = BasicBlock(2*inplane, inplane)
		def forward(self, x):
				out_upper = self.upper(x)
				out_lower = self.downSample(x)
				out_lower = self.low1(out_lower)
				out_lower = self.low2(out_lower)
				out_lower = self.low3(out_lower)
				out_lower = self.upSample(out_lower)
				out = out_lower + out_upper
				return out

class constructShading(nn.Module):
		def __init__(self):
				super(constructShading, self).__init__()
				self.att = Variable(torch.Tensor(np.pi*np.array([1, 2.0/3, 0.25])).cuda()).float()
				self.help_1 = Variable(torch.Tensor([np.sqrt(1.0/(4*np.pi))]).cuda()).float()
				self.help_2 = Variable(torch.Tensor([np.sqrt(3.0/(4*np.pi))]).cuda()).float()
				self.help_3 = Variable(torch.Tensor([np.sqrt(5.0/(4*np.pi))]).cuda()).float()
				self.help_4 = Variable(torch.Tensor([np.sqrt(5.0/(12*np.pi))]).cuda()).float()
				self.help_5 = Variable(torch.Tensor([np.sqrt(5.0/(48*np.pi))]).cuda()).float()
				#self.att = list(np.pi*np.array([1, 2.0/3, 0.25]))
				#self.help_1 = np.sqrt(1.0/(4*np.pi))
				#self.help_2 = np.sqrt(3.0/(4*np.pi))
				#self.help_3 = np.sqrt(5.0/(4*np.pi))
				#self.help_4 = np.sqrt(5.0/(12*np.pi))
				#self.help_5 = np.sqrt(5.0/(48*np.pi))
		def getSH(self, normal):
				'''
					from normal to SH basis
					supposing normal has the format:
					N x X x Y x 3
				'''
				begin_time = time.time()
				numImages = normal.size()[0]
				num_x = normal.size()[1]
				num_y = normal.size()[2]

				x = torch.unsqueeze(normal[:,:,:,0], -1)
				y = torch.unsqueeze(normal[:,:,:,1], -1)
				z = torch.unsqueeze(normal[:,:,:,2], -1)

				# for normals with all elements being 0
				# we set it to be zero
				tmpOnes = x**2 + y**2 + z**2

				#sh_0 = Variable((numImages, num_x, num_y, 1).cuda()).float()*self.att[0]*self.help_1
				sh_0 = tmpOnes.float()*self.att[0]*self.help_1
				sh_1 = self.att[1]*self.help_2*z
				sh_2 = self.att[1]*self.help_2*x
				sh_3 = self.att[1]*self.help_2*y
				sh_4 = self.att[2]*0.5*self.help_3*(2*z*z-x*x-y*y)
				sh_5 = self.att[2]*(3.0*self.help_4)*(x*z)
				sh_6 = self.att[2]*(3.0*self.help_4)*(y*z)
				sh_7 = self.att[2]*(3.0*self.help_5)*(x*x-y*y)
				sh_8 = self.att[2]*(3.0*self.help_4)*(x*y)
				sh = torch.cat((sh_0, sh_1, sh_2, sh_3, sh_4, sh_5, sh_6, sh_7, sh_8), -1)
				return sh

		def forward(self, normal, lighting):
				'''
					get shading based on normal and lighting
				'''
				begin_time = time.time()
				# conver channel to the last dimension
				tmp_normal = normal.permute(0,2,3,1)
				sh = self.getSH(tmp_normal)

				# recover lighting
				light_0 = lighting[:, 0:9]
				light_1 = lighting[:, 9:18]
				light_2 = lighting[:, 18:27]

				numImages = normal.shape[0]
				num_x = normal.shape[2]
				num_y = normal.shape[3]
				shading = Variable(torch.zeros(numImages, num_x, num_y, 3).cuda()).float()
				for i in range(numImages):
						shading[i,:,:,0] = torch.matmul(sh[i,:,:,:], light_0[i])
						shading[i,:,:,1] = torch.matmul(sh[i,:,:,:], light_1[i])
						shading[i,:,:,2] = torch.matmul(sh[i,:,:,:], light_2[i])
				return shading.permute(0, 3, 1, 2) 


class HourglassNet(nn.Module):
    '''
    	basic idea: low layers are shared, upper layers are different	
    	lighting should be estimated from the inner most layer
    '''
    def __init__(self, numBasis):
        super(HourglassNet, self).__init__()
        self.HG2_nFilter = 64
        self.HG1_nFilter = 128 
        self.HG0_nFilter = 256
        
        # split the bottle neck layer into albedo, normal and lighting
        # albedo: 124
        # normal: 124
        # light: 8
        self.albedo_nDim = 124
        self.normal_nDim = 124
        self.light_nDim = 8

        # input to this layer includes: 
        # image: 3 channels
        # albedo: 3 channels
        # normal: 3 channels
        # shading: 3 channels
        self.conv1 = nn.Conv2d(12, self.HG2_nFilter, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(self.HG2_nFilter)
        
        # HG2
        self.HG2_up_albedo = BasicBlock(self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_up_normal = BasicBlock(self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_downSample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.HG2_upSample_albedo = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG2_upSample_normal = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG2_low1 = BasicBlock(self.HG2_nFilter, 2*self.HG2_nFilter)
        self.HG2_low2_normal = BasicBlock(2*self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_low2_albedo = BasicBlock(2*self.HG2_nFilter, self.HG2_nFilter)
        
        # HG1
        self.HG1_up_albedo = BasicBlock(self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_up_normal = BasicBlock(self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_downSample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.HG1_upSample_albedo = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG1_upSample_normal = nn.Upsample(scale_factor=2, mode='bilinear')

        # after this layer, split into albedo, normal, and lighting
        self.HG1_low1 = BasicBlock(self.HG1_nFilter, 2*self.HG1_nFilter)
        self.HG1_low2_normal = BasicBlock(self.normal_nDim, self.HG1_nFilter)
        self.HG1_low2_albedo = BasicBlock(self.albedo_nDim, self.HG1_nFilter)

        self.inner_normal = BasicBlock(self.normal_nDim, self.normal_nDim)
        self.inner_albedo = BasicBlock(self.albedo_nDim, self.albedo_nDim)
        self.inner_light = BasicBlock(self.light_nDim, self.light_nDim)
        
        
        #------------------------------------------------------------------- 
        # for albedo and normal layer, add three convolutional layers to 
        # predict the final result
        #------------------------------------------------------------------- 
        # albedo layer
        self.albedo_conv_1 = nn.Conv2d(self.HG2_nFilter, self.HG2_nFilter/2, kernel_size=3, stride=1, padding=1)
        self.albedo_bn_1 = nn.BatchNorm2d(self.HG2_nFilter/2)
        self.albedo_conv_2 = nn.Conv2d(self.HG2_nFilter/2, 3, kernel_size=3, stride=1, padding=1)
        self.albedo_bn_2 = nn.BatchNorm2d(3)
        self.albedo_conv_3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # normal layer
        self.normal_conv_1 = nn.Conv2d(self.HG2_nFilter, self.HG2_nFilter/2, kernel_size=3, stride=1, padding=1)
        self.normal_bn_1 = nn.BatchNorm2d(self.HG2_nFilter/2)
        self.normal_conv_2 = nn.Conv2d(self.HG2_nFilter/2, 3, kernel_size=3, stride=1, padding=1)
        self.normal_bn_2 = nn.BatchNorm2d(3)
        self.normal_conv_3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # need to construct shading here
        self.getShading = constructShading()
        
        # for lighting
        self.light_conv_1 = nn.Conv2d(self.light_nDim, 64, kernel_size=3, stride=2, padding=1)
        self.light_bn_1 = nn.BatchNorm2d(64)
        self.light_conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.light_bn_2 = nn.BatchNorm2d(128)

        self.light_avePool = nn.AvgPool2d((2,2))
        self.light_FC1 = nn.Linear(128, numBasis) 
    
    def forward(self, x, gl_normal):
        feat = self.conv1(x)
        feat = F.relu(self.bn1(feat))
        
        # get the inner most
        feat_2 = self.HG2_low1(self.HG2_downSample(feat))
        feat_1 = self.HG1_low1(self.HG1_downSample(feat_2))
        # split bottle neck layer into albedo, normal and lighting
        feat_albedo_1 = feat_1[:,0:self.albedo_nDim, :, :]
        feat_normal_1 = feat_1[:,self.albedo_nDim: self.albedo_nDim + self.normal_nDim, :, :]
        feat_light_1 = feat_1[:,self.albedo_nDim + self.normal_nDim:, :, :]
        feat_inner_albedo = self.inner_albedo(feat_albedo_1)
        feat_inner_normal = self.inner_normal(feat_normal_1)
        feat_inner_light = self.inner_light(feat_light_1)
        
        # get albedo
        feat_albedo_1 = self.HG1_upSample_albedo(self.HG1_low2_albedo(feat_inner_albedo)) \
        				+ self.HG1_up_albedo(feat_2)
        feat_albedo_2 = self.HG2_upSample_albedo(self.HG2_low2_albedo(feat_albedo_1)) \
        				+ self.HG2_up_albedo(feat)
        albedo = F.relu(self.albedo_bn_1(self.albedo_conv_1(feat_albedo_2)))
        albedo = F.relu(self.albedo_bn_2(self.albedo_conv_2(albedo)))
        # do not use sigmoid since this is used to model the difference
        albedo = self.albedo_conv_3(albedo)

        
        # get normal
        feat_norm_1 = self.HG1_upSample_normal(self.HG1_low2_normal(feat_inner_normal)) \
        				+ self.HG1_up_normal(feat_2)
        feat_norm_2 = self.HG2_upSample_normal(self.HG2_low2_normal(feat_norm_1)) \
        				+ self.HG2_up_normal(feat)
        normal = F.relu(self.normal_bn_1(self.normal_conv_1(feat_norm_2)))
        normal = F.relu(self.normal_bn_2(self.normal_conv_2(normal)))
        # do not use tanh and normalize normal  
        # since this is used to model the difference
        normal = self.normal_conv_3(normal)
        
        # get lighting
        light_feat = F.relu(self.light_bn_1(self.light_conv_1(feat_inner_light)))
        light_feat = F.relu(self.light_bn_2(self.light_conv_2(light_feat)))
        light_feat = self.light_avePool(light_feat).view(-1, 128)
        lighting = self.light_FC1(light_feat)
        
        # get shading
        # normal used to compute shading is predicted normal + coarse normal
        out_norm = normal + gl_normal
        out_norm = F.normalize(out_norm, p=2, dim=1)
        shading = self.getShading(out_norm, lighting)
        
        return albedo, normal, shading, lighting

if __name__ == '__main__':
		net = HourglassNet()
		x = Variable(torch.Tensor(1, 3, 16, 16))
		albedo, norm, lighting, shading = net(x)
		g = make_dot(shading)
		g.render('HourglassNet')
