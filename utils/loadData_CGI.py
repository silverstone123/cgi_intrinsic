import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import skimage
import random
#import imageio
import skimage.morphology as mpy
from skimage.morphology import square
random.seed(0)
import cv2

class CGI(Dataset):
    '''
    	loading IIW data sets
    '''
    
    def __init__(self, dataFolder, albedoFolder, shadingFolder, normalFolder, maskFolder, fileListName, missingListName, transform=None):
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
        self.missingList = []
        with open(missingListName) as f:
            for line in f:
                self.missingList.append(line.strip())
        
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
		# load image
        imgName = os.path.join(self.dataFolder, fileName + '_mlt.png')
        image = io.imread(imgName)
        if len(image.shape)==2:
            image = np.tile(image[...,None], (1, 3))
        image = np.float32(image)/255.0
       
        
        # load albedo
        albedoName = os.path.join(self.albedoFolder, fileName + '_mlt_albedo.png')
        albedo = io.imread(albedoName)
        if len(albedo.shape)==2:
            albedo = np.tile(albedo[...,None], (1, 3))
        albedo = np.float32(albedo)/255.0
        albedo[albedo < 1e-6] = 1e-6

        # --------------------------------------------------------------------
        # complicated code copied from CGI 
        # I don't really think this block of code is totally correct
        # get shading and mask according to the code
        maskName = os.path.join(self.dataFolder, fileName + "_mlt_mask.png")
        mask = io.imread(maskName)
        mask = np.float32(mask)/255.0

        gt_R_gray = np.mean(albedo, 2)
        mask[gt_R_gray < 1e-6] = 0 
        mask[np.mean(image,2) < 1e-6] = 0
        mask = skimage.morphology.binary_erosion(mask, square(11))
        mask = np.expand_dims(mask, axis = 2)
        mask = np.repeat(mask, 3, axis= 2)
        albedo[albedo < 1e-6] = 1e-6

        rgb_img = image**2.2
        shading = rgb_img / albedo

        #mask[gt_S > 10] = 0 # CGI code this value is set to be 10, but I think it is wrong
        mask[shading > 20] = 0
        mask[shading < 1e-4] = 0

        shading[shading < 1e-4] = 1e-4
        shading[shading > 20] = 20

        if np.sum(mask) < 10:
            max_S = 1.0
        else:
            max_S = np.percentile(shading[mask > 0.5], 90)

        shading = shading/max_S
        mask = np.float32(np.abs(np.sum(mask, axis=2)/3.0 - 1.0)<1e-6)
        #------------------------------------------------------------------------

        
        ## shading saved as raw
        #shadingName = os.path.join(self.shadingFolder, fileName + '.tiff')
        #shading = imageio.imread(shadingName)
        #if len(shading.shape)==2:
        #    shading = np.tile(shading[...,None], (1, 3))
        #shading = shading/20.0
        
        if fileName in self.missingList:
            # no normal
            imgHeight = image.shape[0]
            imgWidth = image.shape[1]
            normal = np.zeros((imgHeight, imgWidth, 3))
            normalMask = np.zeros((imgHeight, imgWidth))
        else:
            normalName = os.path.join(self.normalFolder, fileName + '_norm_camera.png')
            normal = io.imread(normalName)
            normalMaskName = os.path.join(self.normalFolder, fileName + '_valid.png')
            normalMask = io.imread(normalMaskName)

        
        if self.transform:
            image, albedo, shading, normal, mask, normalMask = \
                    self.transform([image, albedo,  shading, normal, mask, normalMask])
        return image, albedo, shading, normal, mask, normalMask

class testTransfer(object):
    def __init__(self, output_size=64):
        # we need to think about this latter
        self.size=output_size
    def __call__(self, sample):
        # center crop
        image, albedo, shading, normal, mask, normalMask = sample

        # directly resize the image
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        albedo = cv2.resize(albedo, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        shading = cv2.resize(shading, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        normal = cv2.resize(normal, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = np.expand_dims(mask, axis=-1)
        normalMask = cv2.resize(normalMask, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        normalMask = np.expand_dims(normalMask, axis=-1)
        
        normal = normal.astype(np.float)
        normal = (normal/255.0-0.5)*2
        normal = normal/(np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
        mask = 1.0*mask/255.0
        normalMask = 1.0*normalMask/255.0
        #mask = mask*normalMask
        
        return image, albedo, shading, normal, mask, normalMask


class cropImg(object):
    '''
        randomly flip, resize and crop
    '''
    def __init__(self, output_size=256, maxSize=300):
        self.size = output_size
        self.maxSize = maxSize
    def __call__(self, sample):
        image, albedo, shading, normal, mask, normalMask= sample

        # randomly resize the image to 256 to 300 images
        imgSize = np.random.randint(self.size, self.maxSize)

        image = cv2.resize(image, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)
        albedo = cv2.resize(albedo, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)
        shading = cv2.resize(shading, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)
        normal = cv2.resize(normal, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)
        normalMask = cv2.resize(normalMask, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)

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
        normalMask = normalMask[sH:sH+self.size, sW:sW+self.size]
        mask = np.expand_dims(mask, -1)
        normalMask = np.expand_dims(normalMask, -1)
        
        #mask = mask*normalMask
        
        # convert to 0-1
        normal = normal.astype(np.float)
        normal = (normal/255.0 - 0.5)*2
        normal = normal/(np.tile(np.linalg.norm(normal, axis=-1, keepdims=True), (1,1,3)) + 1e-6)
        
        mask = 1.0*mask/255.0
        normalMask = 1.0*normalMask/255.0
        return image, albedo, shading, normal, mask, normalMask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, albedo, shading, normal, mask, normalMask = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        albedo = albedo.transpose((2, 0, 1))
        shading = shading.transpose((2, 0, 1))
        normal = normal.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        normalMask = normalMask.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(albedo), \
            torch.from_numpy(shading), torch.from_numpy(normal), \
            torch.from_numpy(mask), torch.from_numpy(normalMask)
