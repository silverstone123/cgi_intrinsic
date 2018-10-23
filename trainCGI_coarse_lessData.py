import sys
sys.path.append('utils')
sys.path.append('model')

import os
import numpy as np
import time

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
from tensorboardX import SummaryWriter

# load module defined by myself
from defineHourglass_64 import *
from utils_shading import *
from defineCriterion import *
#from loadData_tonemapping import *
from loadData_CGI import *
from defineHelp import *
from defineHelp_lighting import *


# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


databaseFolder = '/net/acadia7a/data/xiangyu/hao_intrinsic/intrinsics_final'
listFolder = './list'

criterion = mask_loss()

my_network = HourglassNet(27)
my_network.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer = constructShading()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()


transformer = transforms.Compose([cropImg(output_size=64), ToTensor()])
transformer_test = transforms.Compose([testTransfer(), ToTensor()])

trainLoaderHelper = CGI(dataFolder = os.path.join(databaseFolder, 'images'), 
	albedoFolder=os.path.join(databaseFolder, 'images'),
	shadingFolder=os.path.join(databaseFolder, 'shading'),
	normalFolder=os.path.join(databaseFolder, 'normal'),
	maskFolder=os.path.join(databaseFolder, 'mask'),
	fileListName=os.path.join(listFolder, 'training.list'),
	#fileListName=os.path.join(baseFolder, 'debug.list'),
	missingListName=os.path.join(listFolder, 'missingData.list'),
	transform = transformer)

testLoaderHelper = CGI(dataFolder=os.path.join(databaseFolder, 'images'),
	albedoFolder=os.path.join(databaseFolder, 'images'),
	shadingFolder=os.path.join(databaseFolder, 'shading'),
	normalFolder=os.path.join(databaseFolder, 'normal'),
	maskFolder=os.path.join(databaseFolder, 'mask'),
	fileListName=os.path.join(listFolder, 'validation.list'),
	missingListName=os.path.join(listFolder, 'missingData.list'),
	transform = transformer_test)

trainLoader = torch.utils.data.DataLoader(trainLoaderHelper, 
	batch_size=50, shuffle=True, num_workers=5)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
	batch_size=20, shuffle=False, num_workers=5)

# get the normal/direction of positive and negative spheres
# used to constrain lighting to be positive
sphereDirection_pos, sphereDirection_neg, sphereMask = getSHNormal(64)
sphereDirection_pos = Variable(torch.from_numpy(sphereDirection_pos).cuda()).float()
sphereDirection_neg = Variable(torch.from_numpy(sphereDirection_neg).cuda()).float()
sphereMask = Variable(torch.from_numpy(sphereMask).cuda()).float()

def getLoss(output_albedo, output_shading, output_normal, albedo, shading, normal, mask, normalMask):
    '''
    	compute the loss for the mini-batch
        contains: l1 loss for albedo, normal and shading
                  and their gradient
    '''
    output_albedo_grad = gradientLayer.forward(output_albedo)
    albedo_grad = gradientLayer.forward(albedo)
    output_shading_grad = gradientLayer.forward(output_shading)
    shading_grad = gradientLayer.forward(shading)
    output_normal_grad = gradientLayer.forward(output_normal)
    normal_grad = gradientLayer.forward(normal)
    
    imgMask = mask.expand(-1, 3, -1, -1)
    gradMask = mask.expand(-1, 6, -1, -1)

    imgNormalMask = normalMask.expand(-1, 3, -1, -1)
    gradNormalMask = normalMask.expand(-1, 6, -1, -1)
    
    loss_albedo = criterion(imgMask, torch.abs(albedo - output_albedo))
    loss_shading = criterion(imgMask, torch.abs(shading - output_shading))
    
    loss_albedo_grad = criterion(gradMask, torch.abs(albedo_grad - output_albedo_grad))
    loss_shading_grad = criterion(gradMask, torch.abs(shading_grad - output_shading_grad))
    
    loss_normal = criterion(imgNormalMask, torch.abs(output_normal - normal))
    loss_normal_grad = criterion(gradNormalMask, torch.abs(normal_grad - output_normal_grad))
    
    return loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad

def networkForward(data, Testing=False):
    '''
        given data, compute the loss for the network
        return loss in a list
    '''
    # get output from the network
    inputs, albedo, shading, normal, mask, normalMask = data
    inputs, albedo, shading, normal, mask, normalMask = \
            Variable(inputs.cuda(), volatile=Testing).float(), \
            Variable(albedo.cuda(), volatile=Testing).float(), \
            Variable(shading.cuda(), volatile=Testing).float(), \
            Variable(normal.cuda(), volatile=Testing).float(), \
            Variable(mask.cuda(), volatile=Testing).float(), \
            Variable(normalMask.cuda(), volatile=Testing).float()
    output_albedo, output_normal, output_lighting  = my_network(inputs)
    output_shading = shadingLayer(output_normal, output_lighting)

    # compute the loss
    loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, \
        loss_normal, loss_normal_grad = \
        getLoss(output_albedo, output_shading, output_normal,
            albedo, shading, normal, mask, normalMask)
    
    loss_light_pos = nonNegativeLighting_coarseLevel(samplingLightLayer, sphereDirection_pos, output_lighting)
    loss_light_neg = nonNegativeLighting_coarseLevel(samplingLightLayer, sphereDirection_neg, output_lighting)
    loss_light = loss_light_neg + loss_light_pos


    # append all the losses
    loss = {}
    loss['albedo'] = loss_albedo
    loss['albedo_grad'] = loss_albedo_grad
    loss['shading'] = loss_shading
    loss['shading_grad'] = loss_shading_grad
    loss['normal'] = loss_normal
    loss['normal_grad'] = loss_normal_grad
    loss['light'] = loss_light
    loss['loss_list'] = [loss_albedo.data[0], loss_albedo_grad.data[0], loss_shading.data[0],
        loss_shading_grad.data[0], loss_normal.data[0], loss_normal_grad.data[0], loss_light.data[0]]
    return loss, output_albedo, output_shading, output_normal, albedo, shading, normal


def main(savePath, lr=1e-3, weight_decay=0, total_epoch=100):
    begin_time = time.time()
    print 'learning rate is %.6f' % lr
    print 'weight decay is %.6f' % weight_decay
    print 'epoch is %05d' % total_epoch
    savePath = savePath + '_{:0.4f}_{:0.2f}_{:04d}'.format(lr, weight_decay, total_epoch)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    saveIntermedia = os.path.join(savePath, 'trained_model')
    if not os.path.exists(saveIntermedia):
        os.makedirs(saveIntermedia)
    writer = SummaryWriter(os.path.join(savePath, 'tensorboard'))
    
    monitor_count = 20 # every 20 iterations, output the current loss

    optimizer = optim.Adam(my_network.parameters(), lr = lr, weight_decay=weight_decay)
    fid = open(os.path.join(savePath, 'training.log'), 'w')
    fid_sep = open(os.path.join(savePath, 'training_sep.log'), 'w')
    fid_test = open(os.path.join(savePath, 'testing.log'), 'w')
    fid_test_sep = open(os.path.join(savePath, 'testing_sep.log'), 'w')
    print>>fid_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting'
    print>>fid_test_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting'

    numIte_training = 0
    numIte_testing = 0
    for epoch in range(total_epoch):
        # ---------------------------------------------------------
        # training
        # ---------------------------------------------------------
        running_loss = 0.0
        # loss for each indivisual component
        running_albedo_loss = 0.0
        running_shading_loss = 0.0
        running_normal_loss = 0.0
        running_light_loss = 0.0

        my_network.train(True)
        epoch_time = time.time()
        loss_list = []
        tmp_loss_list = []
        for i , data in enumerate(trainLoader, 0):
            optimizer.zero_grad()
            loss_miniBatch, output_albedo, output_shading, output_normal,\
                    albedo, shading, normal = networkForward(data, Testing=False)
            loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad'] + \
                    loss_miniBatch['shading'] + loss_miniBatch['shading_grad'] + \
                    loss_miniBatch['normal'] + loss_miniBatch['normal_grad'] + loss_miniBatch['light']
            #loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad']
            loss.backward()
            optimizer.step()
            numLoss = len(loss_miniBatch['loss_list'])
            loss_list.append(loss_miniBatch['loss_list'])
            tmp_loss_list.append(loss_miniBatch['loss_list'])

            running_loss += loss.data[0] # [0]
            running_albedo_loss  += loss_miniBatch['albedo'].data[0]
            running_shading_loss  += loss_miniBatch['shading'].data[0]
            running_normal_loss  += loss_miniBatch['normal'].data[0]
            running_light_loss  += loss_miniBatch['light'].data[0]

            if i%monitor_count == monitor_count - 1:
                tmp_loss_list = np.array(tmp_loss_list)
                tmp_loss_list = np.mean(tmp_loss_list, axis=0)
                numIte_training = numIte_training + 1
                print '[%d %5d] loss: ' % (epoch + 1, i+1),
                print '%.4f '*numLoss % tuple(tmp_loss_list)
                print>>fid, '%d %5d ' % (epoch+1, i+1),
                print>>fid, '%.4f '*numLoss % tuple(tmp_loss_list)

                writer.add_scalar('train/loss_albedo', running_albedo_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_shading', running_shading_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_normal', running_normal_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_lighting', running_light_loss/monitor_count, numIte_training)
                writer.add_image('train/albedo', output_albedo)
                writer.add_image('train/shading', output_shading)
                writer.add_image('train/normal', output_normal)
                writer.add_image('train/albedo_gt', albedo)
                writer.add_image('train/shading_gt', shading)
                writer.add_image('train/normal_gt', normal)

                running_loss = 0
                running_albedo_loss = 0
                running_shading_loss = 0
                running_normal_loss = 0
                running_light_loss = 0
                tmp_loss_list = []

        loss_list = np.mean(np.array(loss_list), axis=0)
        print >> fid_sep, '%0.6f '* numLoss % tuple(loss_list)
        print '%0.6f '* numLoss % tuple(loss_list)
        # ---------------------------------------------------------
        # validation 
        # ---------------------------------------------------------
        my_network.train(False)
        test_loss = 0
        # loss for each component
        test_albedo_loss = 0
        test_shading_loss = 0
        test_normal_loss = 0
        test_light_loss = 0

        count = 0
        test_loss_list = []
        for i, data in enumerate(testLoader, 0):
            loss_miniBatch, output_albedo, output_shading, output_normal,\
                    albedo, shading, normal = networkForward(data, Testing=True)
            loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad'] + \
                    loss_miniBatch['shading'] + loss_miniBatch['shading_grad'] + \
                    loss_miniBatch['normal'] + loss_miniBatch['normal_grad'] + loss_miniBatch['light']
            numLoss = len(loss_miniBatch['loss_list'])
            test_loss_list.append(loss_miniBatch['loss_list'])
            test_loss += loss.data[0] #[0]
            test_albedo_loss += loss_miniBatch['albedo']
            test_normal_loss += loss_miniBatch['normal']
            test_shading_loss += loss_miniBatch['shading']
            test_light_loss += loss_miniBatch['light']
            count = count + 1
        test_loss_list = np.mean(np.array(test_loss_list), axis=0)
        print '[%d  ] test loss: ' % (epoch+1),
        print '%.4f '*numLoss % tuple(test_loss_list)
        print>>fid_test, '%d ' % (epoch+1),
        print>>fid_test, '%.4f '*numLoss % tuple(test_loss_list)
        
        writer.add_scalar('test/loss_albedo', test_albedo_loss/count, epoch)
        writer.add_scalar('test/loss_shading', test_shading_loss/count, epoch)
        writer.add_scalar('test/loss_normal', test_normal_loss/count, epoch)
        writer.add_scalar('test/loss_light', test_light_loss/count, epoch)
        writer.add_image('test/albedo', output_albedo)
        writer.add_image('test/shading', output_shading)
        writer.add_image('test/normal', output_normal)
        writer.add_image('test/albedo_gt', albedo)
        writer.add_image('test/shading_gt', shading)
        writer.add_image('test/normal_gt', normal)
    
        print >> fid_test_sep, '%0.6f '* numLoss % tuple(test_loss_list)
        print '%0.6f '* numLoss % tuple(test_loss_list)
    
        tmp_saveName = os.path.join(saveIntermedia, 'trained_model_{:02d}.t7'.format(epoch))
        my_network.cpu()
        torch.save(my_network.state_dict(), tmp_saveName)
        my_network.cuda()
        print 'this epoch cost %s seconds' %(time.time() - epoch_time)
    
    my_network.cpu()
    torch.save(my_network, os.path.join(savePath, 'trained_model.t7'))
    print 'time used for training is %s' % (time.time() - begin_time)
    print('Finished training')
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()
    writer.close()

if __name__ == '__main__':
		savePath = sys.argv[1]
		if not os.path.exists(savePath):
				os.makedirs(savePath)
		if len(sys.argv) > 2:
				lr = float(sys.argv[2])
				weight_decay = float(sys.argv[3])
				total_epoch=int(sys.argv[4])
				main(savePath, lr, weight_decay, total_epoch)
		else:
				main(savePath)
