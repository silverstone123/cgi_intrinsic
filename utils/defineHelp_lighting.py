'''
	contains functions useful for computing lighting
'''

import torch
import numpy as np

def getSHNormal(img_size):
		'''
			get the normal on a sphere, both positive half sphere and negative half sphere
			According to the normal provided by SUNCG
			NOTE: coordinate is in world coordinate
			X: pointing left
			Y: pointing inward
			Z: pointing up
		'''
		i = (1.0*np.arange(img_size) - 1.0*img_size/2)/img_size*2
		j = (1.0*np.arange(img_size) - 1.0*img_size/2)/img_size*2
		X, Z = np.meshgrid(i,j)
		Z = -1*Z
		valid = 1-(X**2 + Z**2) >=0
		ind_zero = np.zeros((img_size, img_size))

		Y = np.sqrt(np.maximum(ind_zero,  1 - (X**2 + Z**2)))  # z value for positive half sphere
		tmp = np.concatenate((X[None,...], -1*Y[None,...], Z[None,...]), axis=0)
		sphereNormal = tmp*np.tile(valid[None,...], (3, 1, 1))
		
		neg_Y = -1*np.sqrt(np.maximum(ind_zero,  1 - (X**2 + Z**2))) # Y value for negative half sphere
		tmp = np.concatenate((X[None,...], neg_Y[None,...], Z[None,...]), axis=0)
		neg_sphereNormal = tmp*np.tile(valid[None,...], (3, 1, 1))
		return sphereNormal[None,...], neg_sphereNormal[None,...], np.tile(valid[None,None,...], (1,1,1,1))*1.0

def nonNegativeLighting_coarseLevel(samplingLayer, directions, lighting):
		'''
			enforce the lighting to be positive on the coarest level
			samplingLayer: nn layer to reconstruct shading
			directions: direction of light (of half sphere)
			lighting: Nx27
			loss = min(0, lighting)**2
		'''
		numSamples = lighting.size()[0]  # number of samples
		shadingH = directions.size()[2]  # Height of half sphere
		shadingW = directions.size()[3]  # width of half sphere
		directions = directions.expand(numSamples, 3, -1, -1)

		ind_zeros = torch.autograd.Variable(torch.zeros(numSamples, 3, shadingH, shadingW).cuda()).float()
		
		shading = samplingLayer(directions, lighting)
		loss = torch.sum(torch.min(ind_zeros, shading)**2)/(numSamples)
		return loss

def nonNegativeLighting_fineLevel(shadingLayer, directions, fineLighting, coarseLighting):
		'''
			enforce the lighting to be positive
			shaingLayer: nn layer to reconstruct shading 
			normal: normal (of half sphere)
			fineLighting: Nx27xWxH
			coarseLighting: Nx27
			lighting = fineLighting + coarseLighting
			loss = min(0, lighting)**2
		'''
		numSamples = fineLighting.size()[0]
		shadingH = directions.size()[2]  # Height of half sphere
		shadingW = directions.size()[3]  # width of half sphere
		normal = directions.expand(numSamples, 3, shadingH, shadingW)

		ind_zeros = torch.autograd.Variable(torch.zeros(numSamples, 3, shadingH, shadingW).cuda()).float()

		H = fineLighting.size()[2]
		W = fineLighting.size()[3]

		loss = torch.autograd.Variable(torch.Tensor([0]).cuda()).float()
		for i in range(H):
			for j in range(W):
					shading = shadingLayer(directions, fineLighting[:, :, i,j] + coarseLighting)
					loss = loss + torch.sum(torch.min(ind_zeros, shading)**2)/(numSamples)
		return loss/(H*W)

def constrainLighting(shadingLayer, normal, fineLighting, coarseLighting):
		'''
			enforce the lighting to be positive
			shaingLayer: nn layer to reconstruct shading
			normal: normal (of half sphere)
			fineLighting: Nx27xWxH
			coarseLighting: Nx27
			lighting = fineLighting + coarseLighting
			loss = min(0, lighting)**2
		'''
		numSamples = fineLighting.size()[0]
		shadingH = normal.size()[2]  # Height of half sphere
		shadingW = normal.size()[3]  # width of half sphere
		normal = normal.expand(numSamples, 3, shadingH, shadingW)

		ind_zeros = torch.autograd.Variable(torch.zeros(numSamples, 3, shadingH, shadingW).cuda()).float()
		ind_ones = torch.autograd.Variable(torch.ones(numSamples, 3, shadingH, shadingW).cuda()).float()

		H = fineLighting.size()[2]
		W = fineLighting.size()[3]

		loss = torch.autograd.Variable(torch.Tensor([0]).cuda()).float()
		for i in range(H):
				for j in range(W):
						shading = shadingLayer(normal, fineLighting[:, :, i,j] + coarseLighting)
						loss = loss + torch.sum(torch.min(ind_zeros, shading)**2)/(numSamples)
						loss = loss + torch.sum((torch.max(ind_ones, shading) - ind_ones)**2)/(numSamples)
		return loss/(H*W)
