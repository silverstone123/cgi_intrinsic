import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/scratch1/intrinsicImage/myProject/code_synthetic/functional-zoo/')
from visualize import make_dot
import time

# we define the convolutional network to deal with 
# a single image.
# the sub-network will work on local patches of the 
# image, and finally, we construct the whole image
# NOTE: there are overlap between local patches
# so we have to define our own pooling layer
# here we use average pooling

def getPatchWeight(H_patchSize, W_patchSize, H_stride, W_stride, 
        boarderUP=False, boarderDOWN=False, boarderLEFT=False, boarderRIGHT=False):
    '''
        this function is used to prepare the patch weight for bilinear interpolation
        H_patchSize: height of the patch
        W_patchSize: width of the patch
        H_stride: stride for height
        W_stride: stride for width
        boarderUP: indicator, whether this patch is on the up boarder of the image: TRUE or FALSE
        boarderDOWN: indicator, whether this patch is on the lower boarder of the image: TRUE or FALSE
        boarderLEFT: indicator, whether this patch is on the left boarder of the image: TRUE or FALSE
        boarderRIGHT: indicator, whether this patch is on the right boarder of the image: TRUE or FALSE
    '''
    # need right: 0 to (W_patchSize - W_stride)
    # need left: W_stride to W_patchSize
    # need down: 0 to (H_patchSize - H_stride)
    # need up: H_stride to H_patchSize
    W_length = W_patchSize - W_stride
    H_length = H_patchSize - H_stride
    patchWeight = np.zeros((H_patchSize, W_patchSize))
    for i in range(H_patchSize):
        for j in range(W_patchSize):
            if j < W_patchSize - W_stride and (not boarderLEFT):
                # contribute as right
                W_dist = j
            elif j > W_stride and (not boarderRIGHT):
                # contribute as left
                W_dist = W_patchSize - j
            else: 
                # no contribution to width
                W_dist = W_length
            if i < H_patchSize - H_stride and (not boarderUP):
                # contribute as down
                H_dist = i
            elif i > H_stride and (not boarderDOWN):
                # contribute as up
                H_dist = H_patchSize - i
            else:
                # no contribution to height
                H_dist = H_length
            patchWeight[i, j] = 1.0*H_dist*W_dist/(W_length*H_length)
    return patchWeight



class myConv_subNetwork(nn.Module):
    def __init__(self, H_patchSize, W_patchSize, H_stride, W_stride, H_size, W_size):
        super(myConv_subNetwork, self).__init__()
        self.H_patchSize = H_patchSize
        self.W_patchSize = W_patchSize
        self.H_stride = H_stride
        self.W_stride = W_stride
        self.H_size = H_size
        self.W_size = W_size
        
        self.numW = (W_size - (W_patchSize - H_stride))/H_stride
        self.numH = (H_size - (H_patchSize - H_stride))/H_stride
        
        self.numPatches = self.numW * self.numH
        self.tmpOnesPatch = Variable(torch.ones(H_patchSize, W_patchSize).cuda()).float()
        
        self.patchWeight = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride)).cuda()).float()
        self.patchWeightUL = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderUP=True, boarderLEFT=True)).cuda()).float()
        self.patchWeightU = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderUP=True)).cuda()).float()
        self.patchWeightUR= Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderUP=True, boarderRIGHT=True)).cuda()).float()
        self.patchWeightR = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderRIGHT=True)).cuda()).float()
        self.patchWeightDR = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderRIGHT=True, boarderDOWN=True)).cuda()).float()
        self.patchWeightDL = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderLEFT=True, boarderDOWN=True)).cuda()).float()
        self.patchWeightD = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderDOWN=True)).cuda()).float()
        self.patchWeightL = Variable(torch.Tensor(getPatchWeight(H_patchSize, W_patchSize, 
            H_stride, W_stride, boarderLEFT=True)).cuda()).float()
    
    def getLoss(self, patch_1, patch_2):
        # get the l1 smooth term for two patches
        numSamples = patch_1.size()[0]
        numChannel = patch_1.size()[1]
        row = patch_1.size()[2]
        col = patch_1.size()[3]
        return torch.sum(torch.abs(patch_1 - patch_2))/(numSamples*numChannel*row*col)
    
    def forward(self, inputTensor, inputNorm, subNetwork):
        '''
            subNetwork is the network that will deal with each local patches
        '''
        
        numSamples = inputTensor.shape[0]
        numChannel = inputTensor.shape[1]
        numHeight = inputTensor.shape[2]
        numWidth = inputTensor.shape[3]
        output_albedo = Variable(torch.zeros(numSamples, 3, numHeight, numWidth).cuda()).float()
        output_normal = Variable(torch.zeros(numSamples, 3, numHeight, numWidth).cuda()).float()
        output_shading = Variable(torch.zeros(numSamples, 3, numHeight, numWidth).cuda()).float()
        output_lighting = Variable(torch.zeros(numSamples, 27, self.numH, self.numW).cuda()).float()
        
        #-------------------------------------------------------------
        #   second inplementation method, for loop prepare the data
        #   forward just once, a little bit faster than the above method
        #   but still slow
        #-------------------------------------------------------------
        batch_input = Variable(torch.zeros(numSamples*self.numPatches, numChannel,
            self.H_patchSize, self.W_patchSize).cuda()).float()
        norm_input = Variable(torch.zeros(numSamples*self.numPatches, 3,
            self.H_patchSize, self.W_patchSize).cuda()).float()
        count = 0

        begin_time = time.time()
        for i in range(self.numH):
            for j in range(self.numW):
                start_x = i*self.H_stride
                end_x = start_x + self.H_patchSize
                start_y = j*self.W_stride
                end_y = start_y + self.W_patchSize
                tmp_input = inputTensor[:, :, start_x:end_x, start_y:end_y]
                tmp_norm = inputNorm[:,:,start_x:end_x, start_y:end_y]
                batch_input[count*numSamples:(count+1)*numSamples] = tmp_input
                norm_input[count*numSamples:(count+1)*numSamples] = tmp_norm
                count = count + 1
        #print 'time used to prepare data is %s' % (time.time() - begin_time)
        #print 'times used for check point 1 is %s ' % (time.time() - begin_time)
        begin_time = time.time()
        batch_output = subNetwork(batch_input, norm_input)
        #print 'time used for forward pass is %s' % (time.time() - begin_time)
        #print 'times used for check point 2 is %s ' % (time.time() - begin_time)
        
        # we also want to compute the loss for overlapping regions
        # for each patch, we compute the l1-distance of the overlapping part
        # w.r.t. the up and left patch for shading
        overlap_loss = Variable(torch.Tensor([0]).cuda()).float()
        
        begin_time = time.time()
        count = 0
        # i*numW + j = count
        for i in range(self.numH):
            for j in range(self.numW):

                if i == 0 and j == 0:
                    tmp_patchWeight = self.patchWeightUL
                elif i == 0 and j == self.numW-1:
                    tmp_patchWeight = self.patchWeightUR
                elif i == self.numH -1 and j == 0:
                    tmp_patchWeight = self.patchWeightDL
                elif i == self.numH - 1 and j == self.numW - 1:
                    tmp_patchWeight = self.patchWeightDR
                elif i == 0:
                    tmp_patchWeight = self.patchWeightU
                elif j == 0:
                    tmp_patchWeight = self.patchWeightL
                elif i == self.numW - 1:
                    tmp_patchWeight = self.patchWeightD
                elif j == self.numW - 1:
                    tmp_patchWeight = self.patchWeightR
                else:
                    tmp_patchWeight = self.patchWeight
                
                start_x = i*self.H_stride
                end_x = start_x + self.H_patchSize
                start_y = j*self.W_stride
                end_y = start_y + self.W_patchSize
                output_albedo[:,:,start_x:end_x, start_y:end_y] = \
                    output_albedo[:,:,start_x:end_x, start_y:end_y] +\
                    batch_output[0][count*numSamples:(count+1)*numSamples] * tmp_patchWeight
                output_normal[:,:,start_x:end_x, start_y:end_y] = \
                    output_normal[:,:,start_x:end_x, start_y:end_y] + \
                    batch_output[1][count*numSamples:(count+1)*numSamples] * tmp_patchWeight
                current_shading = batch_output[2][count*numSamples:(count+1)*numSamples]
                output_shading[:,:,start_x:end_x, start_y:end_y] = \
                    output_shading[:,:,start_x:end_x, start_y:end_y] + \
                    current_shading * tmp_patchWeight
                if j - 1 >= 0:
                    # it has one left patch
                    output_patch = current_shading[:,:,:,0:self.W_patchSize - self.W_stride]
                    tmp_count = i*self.numW + (j-1)
                    tmp_shading = batch_output[2][tmp_count*numSamples:(tmp_count+1)*numSamples]
                    tmp_patch = tmp_shading[:,:,:,self.W_stride:]
                    overlap_loss = overlap_loss + self.getLoss(output_patch, tmp_patch)
                if i - 1 >= 0:
                    # it has one up patch
                    output_patch = current_shading[:,:,0:self.H_patchSize - self.H_stride,:]
                    tmp_count = (i-1)*self.numW + j
                    tmp_shading = batch_output[2][tmp_count*numSamples:(tmp_count+1)*numSamples]
                    tmp_patch = tmp_shading[:,:,self.H_stride:,:]
                    overlap_loss = overlap_loss + self.getLoss(output_patch, tmp_patch)
                
                output_lighting[:,:,i, j] = \
                				batch_output[3][count*numSamples:(count+1)*numSamples]
                count = count + 1

        #print 'time used to combine results is %s' % (time.time() - begin_time)
        return output_albedo, output_normal, output_shading, output_lighting, overlap_loss/count
