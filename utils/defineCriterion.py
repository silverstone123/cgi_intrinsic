import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class MSELoss(nn.Module):
    def __init__(self):
    	super(MSELoss, self).__init__()
    def forward(self, groundTruth, output):
    	# compute MSE loss
    	# groundTruth and output are tensors with same size:
    	# NxCxHxW
    
    	if groundTruth.size()[0] != output.size()[0] or \
    		groundTruth.size()[1] != output.size()[1] or \
    		groundTruth.size()[2] != output.size()[2] or \
    		groundTruth.size()[3] != output.size()[3]:
    			print(groundTruth.size())
    			print(output.size())
    			raise ValueError('size of groud truth and ouptut does not match')
    
    	# numData = output.size()[0]
    	# total_loss = Variable(torch.cuda.FloatTensor([0]))
    
    	#for i in range(numData):
    	#	tmp_1 = groundTruth[i]
    	#	tmp_2 = output[i]
    	#	tmp = (tmp_1 - tmp_2)**2
    	#	loss = torch.sum(tmp)/(output.size()[1]*output.size()[2]*output.size()[3])
    	#	total_loss += loss
    	# return total_loss/numData
        return torch.sum((groundTruth - output)**2) \
                /(output.size()[0]*output.size()[1]*output.size()[2]*output.size()[3])

class mask_loss(nn.Module):
    '''
    	used for the first order term in augmented Lagrangian method
    '''
    def __init__(self):
    		super(mask_loss, self).__init__()
    def forward(self, mask, output):
    	# compute MSE loss
    	# groundTruth and output are tensors with same size:
    	# NxCxHxW
    
        if mask.size()[1] != output.size()[1] or \
        	mask.size()[2] != output.size()[2] or \
        	mask.size()[3] != output.size()[3]:
        		print(mask.size())
        		print(output.size())
        		raise ValueError('size of groud truth and ouptut does not match')
    
    	#numData = output.size()[0]
    	#total_loss = Variable(torch.cuda.FloatTensor([0]))
    	#channel = output.size()[1]
    	#for i in range(numData):
    	#	tmpOutput = output[i]
    	#	tmpMask = mask[i]
    	#	tmp = tmpMask*tmpOutput
    	#	#if np.isnan(tmp.sum().data.cpu().numpy()):
    	#	#		print 'tmp has nan'
    	#	#		raise
    	#	#if np.isnan(tmpMask.sum().data.cpu().numpy()):
    	#	#		print 'mask has none'
    	#	#		raise
    	#	loss = tmp.sum()/(tmpMask.sum() + 1e-6)
    	#	total_loss += loss
        #return total_loss/numData
		
        total_loss = (mask * output).sum()/(mask.sum() + 1e-6)
        return total_loss

class SiMSELoss(nn.Module):
	def __init__(self):
		super(SiMSELoss, self).__init__()
		self.eps = 1e-10
	def forward(self, groundTruth, output):
		# compute SiMSELoss according to the paper
		# get an alpha to minimize 
		#     (groundTruth - alpha*output)^2
		# then the loss is defined as the above 
		
		if groundTruth.size()[0] != output.size()[0] or \
			groundTruth.size()[1] != output.size()[1] or \
			groundTruth.size()[2] != output.size()[2] or \
			groundTruth.size()[3] != output.size()[3]:
				print(groundTruth.size())
				print(output.size())
				raise ValueError('size of groud truth and ouptut does not match')

		numData = output.size()[0]
		alpha = torch.sum(groundTruth*output)/(torch.sum(output**2) + self.eps)
		return torch.sum((groundTruth - alpha*output)**2) \
                /(output.size()[1]*output.size()[2]*output.size()[3]*numData)
