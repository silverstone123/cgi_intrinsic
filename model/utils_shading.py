import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
# inplemenet layers that are usefull

class SHFromBasis(nn.Module):
    '''
        represent lighting using basis
        i.e. using lighting prior to get the base
            lighting can be represented by these base
    '''
    def __init__(self, lightDirection, numBasis):
        super(SHFromBasis, self).__init__()
        self.help_1 = Variable(torch.Tensor([np.sqrt(1.0/(4*np.pi))]).cuda()).float()
        self.help_2 = Variable(torch.Tensor([np.sqrt(3.0/(4*np.pi))]).cuda()).float()
        self.help_3 = Variable(torch.Tensor([np.sqrt(5.0/(4*np.pi))]).cuda()).float()
        self.help_4 = Variable(torch.Tensor([np.sqrt(5.0/(12*np.pi))]).cuda()).float()
        self.numBasis = numBasis
        
        # lightDirection is a matrix of Nx3, 
        # which contrains N lighting directions
        x = torch.unsqueeze(lightDirection[:,0], 0)
        y = torch.unsqueeze(lightDirection[:,1], 0)
        z = torch.unsqueeze(lightDirection[:,2], 0)
        
        tmpOnes = x**2 + y**2 + z**2
        sh_0 = tmpOnes.float()*self.help_1
        sh_1 = self.help_2*z
        sh_2 = self.help_2*x
        sh_3 = self.help_2*y
        sh_4 = 0.5*self.help_3*(3*z**2 - 1)
        sh_5 = 3.0*self.help_4*(x*z)
        sh_6 = 3.0*self.help_4*(y*z)
        sh_7 = 1.5*self.help_4*(x**2 - y**2)
        sh_8 = 3.0*self.help_4*(x*y)
        
        self.sh = torch.cat((sh_0, sh_1, sh_2, sh_3, sh_4, sh_5, sh_6, sh_7, sh_8), 0)

    def forward(self, light_parameters):
        # get SH based on the parameters
        # supposing the parameters are R, G, B 
        numLight = light_parameters.shape[0]
        
        light_R = light_parameters[:,0:self.numBasis]
        light_G = light_parameters[:,self.numBasis:2*self.numBasis]
        light_B = light_parameters[:,2*self.numBasis:3*self.numBasis]
        
        light = Variable(torch.zeros(numLight, 27).cuda()).float()
        light[:, 0:9] = torch.t(torch.matmul(self.sh, torch.t(light_R)))
        light[:, 9:18] = torch.t(torch.matmul(self.sh, torch.t(light_G)))
        light[:, 18:27] = torch.t(torch.matmul(self.sh, torch.t(light_B)))
        
        return light

class constructShading(nn.Module):
    '''
        this class is used to construct shading based on normal and lighting
    '''
    def __init__(self):
        super(constructShading, self).__init__()
        self.att = Variable(torch.Tensor(np.pi*np.array([1, 2.0/3, 0.25])).cuda()).float()
        self.help_1 = Variable(torch.Tensor([np.sqrt(1.0/(4*np.pi))]).cuda()).float()
        self.help_2 = Variable(torch.Tensor([np.sqrt(3.0/(4*np.pi))]).cuda()).float()
        self.help_3 = Variable(torch.Tensor([np.sqrt(5.0/(4*np.pi))]).cuda()).float()
        self.help_4 = Variable(torch.Tensor([np.sqrt(5.0/(12*np.pi))]).cuda()).float()
        self.help_5 = Variable(torch.Tensor([np.sqrt(5.0/(48*np.pi))]).cuda()).float()
    
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
        # conver channel to the last dimension
        tmp_normal = normal.permute(0,2,3,1)
        sh = self.getSH(tmp_normal)
        
        # recover RGB lighting
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

class constructShading_lightImg(nn.Module):
    '''
        this class is used to construct shading based on normal and lighting
        here we compute per-pixel lighting
    '''
    def __init__(self):
        super(constructShading_lightImg, self).__init__()
        self.att = Variable(torch.Tensor(np.pi*np.array([1, 2.0/3, 0.25])).cuda()).float()
        self.help_1 = Variable(torch.Tensor([np.sqrt(1.0/(4*np.pi))]).cuda()).float()
        self.help_2 = Variable(torch.Tensor([np.sqrt(3.0/(4*np.pi))]).cuda()).float()
        self.help_3 = Variable(torch.Tensor([np.sqrt(5.0/(4*np.pi))]).cuda()).float()
        self.help_4 = Variable(torch.Tensor([np.sqrt(5.0/(12*np.pi))]).cuda()).float()
        self.help_5 = Variable(torch.Tensor([np.sqrt(5.0/(48*np.pi))]).cuda()).float()
    
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
        # conver channel to the last dimension
        tmp_normal = normal.permute(0,2,3,1)
        sh = self.getSH(tmp_normal)
        
        # recover RGB lighting
        lighting = lighting.permute(0,2,3,1)
        light_0 = lighting[:, :, :, 0:9]
        light_1 = lighting[:, :, :, 9:18]
        light_2 = lighting[:, :, :, 18:27]
        
        numImages = normal.shape[0]
        num_x = normal.shape[2]
        num_y = normal.shape[3]
        shading = Variable(torch.zeros(numImages, num_x, num_y, 3).cuda()).float()
        shading[:,:,:,0] = torch.sum(sh*light_0, dim=-1)
        shading[:,:,:,1] = torch.sum(sh*light_1, dim=-1)
        shading[:,:,:,2] = torch.sum(sh*light_2, dim=-1)
        #for i in range(numImages):
        #	shading[i,:,:,0] = torch.matmul(sh[i,:,:,:], light_0[i])
        #	shading[i,:,:,1] = torch.matmul(sh[i,:,:,:], light_1[i])
        #	shading[i,:,:,2] = torch.matmul(sh[i,:,:,:], light_2[i])
        return shading.permute(0, 3, 1, 2) 

class samplingLight(nn.Module):
    '''
        this class is used to sampling light based on spherical harmonics
        NOTE: this class is the same with constructShading, the difference is that we do not
              multiply the basis with attanuation since there is no convolution on the sphere
    '''
    def __init__(self):
        super(samplingLight, self).__init__()
        self.help_1 = Variable(torch.Tensor([np.sqrt(1.0/(4*np.pi))]).cuda()).float()
        self.help_2 = Variable(torch.Tensor([np.sqrt(3.0/(4*np.pi))]).cuda()).float()
        self.help_3 = Variable(torch.Tensor([np.sqrt(5.0/(4*np.pi))]).cuda()).float()
        self.help_4 = Variable(torch.Tensor([np.sqrt(5.0/(12*np.pi))]).cuda()).float()
        self.help_5 = Variable(torch.Tensor([np.sqrt(5.0/(48*np.pi))]).cuda()).float()
    
    def getSH(self, normal):
        '''
        	from normal to SH basis
        	supposing normal has the format:
        	N x X x Y x 3
        '''
        
        x = torch.unsqueeze(normal[:,:,:,0], -1)
        y = torch.unsqueeze(normal[:,:,:,1], -1)
        z = torch.unsqueeze(normal[:,:,:,2], -1)
        
        # for normals with all elements being 0
        # we set it to be zero
        tmpOnes = x**2 + y**2 + z**2
        
        sh_0 = tmpOnes.float()*self.help_1
        sh_1 = self.help_2*z
        sh_2 = self.help_2*x
        sh_3 = self.help_2*y
        sh_4 = 0.5*self.help_3*(2*z*z-x*x-y*y)
        sh_5 = (3.0*self.help_4)*(x*z)
        sh_6 = (3.0*self.help_4)*(y*z)
        sh_7 = (3.0*self.help_5)*(x*x-y*y)
        sh_8 = (3.0*self.help_4)*(x*y)
        sh = torch.cat((sh_0, sh_1, sh_2, sh_3, sh_4, sh_5, sh_6, sh_7, sh_8), -1)
        return sh
    
    def forward(self, directions, lighting):
        '''
        	get shading based on normal and lighting
        '''
        # conver channel to the last dimension
        tmp_directions = directions.permute(0,2,3,1)
        sh = self.getSH(tmp_directions)
        
        # recover RGB lighting
        light_0 = lighting[:, 0:9]
        light_1 = lighting[:, 9:18]
        light_2 = lighting[:, 18:27]
        
        numImages = directions.shape[0]
        num_x = directions.shape[2]
        num_y = directions.shape[3]
        shading = Variable(torch.zeros(numImages, num_x, num_y, 3).cuda()).float()
        for i in range(numImages):
        	shading[i,:,:,0] = torch.matmul(sh[i,:,:,:], light_0[i])
        	shading[i,:,:,1] = torch.matmul(sh[i,:,:,:], light_1[i])
        	shading[i,:,:,2] = torch.matmul(sh[i,:,:,:], light_2[i])
        return shading.permute(0, 3, 1, 2) 
