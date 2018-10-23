import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import random
random.seed(0)
import cv2

class SUNCG(Dataset):
    '''
    	loading IIW data sets
    '''
    
    def __init__(self, dataFolder, albedoFolder, shadingFolder, normalFolder, maskFolder, fileListName, transform=None):
        '''
        	dataFolder: contains images
        	albedoFolder: contains albedo
        	shadingFolder: contains shading
        	normalFolder: contains normal information
        	fileListName: all file names
        '''
        
        self.fileList = []
        with open(fileListName) as f:
            for line in f:
                self.fileList.append(line.strip())
        
        self.dataFolder = dataFolder
        self.albedoFolder = albedoFolder
        self.shadingFolder = shadingFolder
        self.normalFolder = normalFolder
        self.maskFolder = maskFolder
        self.transform = transform
    
    def __len__(self):
        return len(self.fileList)
    
    def __getitem__(self, idx):
        fileName = self.fileList[idx]
        #print fileName
        imgName = os.path.join(self.dataFolder, fileName)
        image = io.imread(imgName)
        if len(image.shape)==2:
            image = np.tile(image[...,None], (1, 3))
        
        albedoName = os.path.join(self.albedoFolder, fileName)
        albedo = io.imread(albedoName)
        if len(albedo.shape)==2:
            albedo = np.tile(albedo[...,None], (1, 3))
        
        shadingName = os.path.join(self.shadingFolder, fileName)
        shading = io.imread(shadingName)
        if len(shading.shape)==2:
            shading = np.tile(shading[...,None], (1, 3))
        
        normalName = os.path.join(self.normalFolder, fileName[0:-8] + '_norm_camera.png')
        normal = io.imread(normalName)
        
        normalMaskName = os.path.join(self.normalFolder, fileName[0:-8] + '_valid.png')
        normalMask = io.imread(normalMaskName)
        
        maskName = os.path.join(self.maskFolder, fileName)
        mask = io.imread(maskName)
        #print 'loading mask'
        #print np.sum(mask)
        
        if self.transform:
            image, albedo, shading, normal, mask = \
                    self.transform([image, albedo,  shading, normal, mask, normalMask])
        return image, albedo, shading, normal, mask

class testTransfer(object):
    def __init__(self, output_size=64):
        # we need to think about this latter
        self.size=output_size
    def __call__(self, sample):
        # center crop
        image, albedo, shading, normal, mask, normalMask = sample
        #H = image.shape[0]
        #W = image.shape[1]
        #maxH = H - self.size
        #maxW = W - self.size
        #sH = int(1.0*maxH/2)
        #sW = int(1.0*maxW/2)
        
        #image = image[sH:sH+self.size, sW:sW+self.size,:]
        #albedo = albedo[sH:sH+self.size, sW:sW+self.size,:]
        #shading = shading[sH:sH+self.size, sW:sW+self.size,:]
        #normal = normal[sH:sH+self.size, sW:sW+self.size,:]
        #mask = mask[sH:sH+self.size, sW:sW+self.size]

        # directly resize the image
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        albedo = cv2.resize(albedo, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        shading = cv2.resize(shading, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        normal = cv2.resize(normal, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = np.expand_dims(mask, axis=-1)
        normalMask = cv2.resize(normalMask, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        normalMask = np.expand_dims(normalMask, axis=-1)
        
        image = 1.0*image/255.0
        albedo = 1.0*albedo/255.0
        shading = 1.0*shading/255.0
        normal = normal.astype(np.float)
        normal = (normal/255.0-0.5)*2
        normal = normal/(np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
        mask = 1.0*mask/255.0
        normalMask = 1.0*normalMask/255.0
        mask = mask*normalMask
        
        return image, albedo, shading, normal, mask


class cropImg(object):
    '''
        expand image first and then crop
    '''
    def __init__(self, output_size=224, expand=10):
        self.size = output_size
        self.expand = 10
    def __call__(self, sample):
        image, albedo, shading, normal, mask, normalMask= sample
        image = cv2.resize(image, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        albedo = cv2.resize(albedo, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        shading = cv2.resize(shading, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        normal = cv2.resize(normal, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        
        # random crop
        H = image.shape[0]
        W = image.shape[1]
        maxH = H - self.size
        maxW = W - self.size
        sH = random.randint(0, maxH)
        sW = random.randint(0, maxW)
        
        image = image[sH:sH+self.size, sW:sW+self.size,:]
        albedo = albedo[sH:sH+self.size, sW:sW+self.size,:]
        shading = shading[sH:sH+self.size, sW:sW+self.size,:]
        normal = normal[sH:sH+self.size, sW:sW+self.size,:]
        mask = mask[sH:sH+self.size, sW:sW+self.size]
        mask = np.expand_dims(mask, axis=-1)
        normalMask = normalMask[sH:sH+self.size, sW:sW+self.size]
        normalMask = np.expand_dims(normalMask, axis=-1)
        
        mask = mask*normalMask
        
        # convert to 0-1
        image = 1.0*image/255.0
        albedo = 1.0*albedo/255.0
        shading = 1.0*shading/255.0
        normal = normal.astype(np.float)
        normal = (normal/255.0 - 0.5)*2
        normal = normal/(np.tile(np.linalg.norm(normal, axis=-1, keepdims=True), (1,1,3)) + 1e-6)
        
        mask = 1.0*mask/255.0
        return image, albedo, shading, normal, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, albedo, shading, normal, mask = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        albedo = albedo.transpose((2, 0, 1))
        shading = shading.transpose((2, 0, 1))
        normal = normal.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(albedo), \
            torch.from_numpy(shading), torch.from_numpy(normal), \
            torch.from_numpy(mask)
