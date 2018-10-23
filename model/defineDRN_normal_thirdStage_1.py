from defineNetwork_Basic import *
from utils_shading import *

# we define dilated residual network based on the paper
#              Dilated Residual Networks
# Fisher Yu, Vladlen Koltun and Thomas Funkhouser
# the code is adapted from
# https://github.com/fyu/drn/blob/master/drn.py

# NOTE: our defined network is fully convolutional, 
# can deal with different resolution of images

class basicNetwork(nn.Module):
    def __init__(self, inplane, outplane):
        super(basicNetwork, self).__init__()
        self.nFilters_basic=16

        self.nFilters_1=32
        self.nFilters_2=32
        self.nFilters_3=64
        self.nFilters_4=32
        #self.nFilters_5=32

        self.dilation_1=1
        self.dilation_2=2
        self.dilation_3=4
        self.dilation_4=1
        self.dilation_5=1

        self.nFilters=32

        self.conv1 = nn.Conv2d(inplane, self.nFilters_basic, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(self.nFilters_basic)

        # shared
        self.block_1= BasicBlock(self.nFilters_basic, self.nFilters_1, stride=1, padding=self.dilation_1, dilation=self.dilation_1)
        self.block_2= BasicBlock(self.nFilters_1, self.nFilters_2, stride=1, padding=2, dilation=self.dilation_2)
        self.block_3= BasicBlock(self.nFilters_2, self.nFilters_3, stride=1, padding=4, dilation=self.dilation_3)
        self.block_4= BasicBlock(self.nFilters_3, self.nFilters_4, stride=1, padding=1, dilation=self.dilation_4)
        #self.block_5= BasicBlock(self.nFilters_4, self.nFilters_5, stride=1, padding=1, dilation=self.dilation_5)

        # fine level, 3 1x1 convolutional layers
        self.fine_conv1 = nn.Conv2d(self.nFilters_basic, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.fine_bn1 = nn.BatchNorm2d(self.nFilters)
        self.fine_conv2 = nn.Conv2d(self.nFilters, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.fine_bn2 = nn.BatchNorm2d(self.nFilters)
        self.fine_conv3 = nn.Conv2d(self.nFilters, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.fine_bn3 = nn.BatchNorm2d(self.nFilters)

        self.pred_1 = nn.Conv2d(self.nFilters, outplane, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.pred_2 = nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

    def forward(self, x):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat_1 = self.block_1(feat)
        feat_2 = self.block_2(feat_1)
        feat_3 = self.block_3(feat_2)
        feat_4 = self.block_4(feat_3) + feat_2 + feat_1
        #feat_5 = self.block_5(feat_4) + feat_1

        fine_1 = F.relu(self.fine_bn1(self.fine_conv1(feat)))
        fine_2 = F.relu(self.fine_bn2(self.fine_conv2(fine_1)))
        fine_3 = F.relu(self.fine_bn3(self.fine_conv3(fine_2)))

        #output = F.relu(self.bn2(self.pred_1(feat_5 + fine_3)))
        output = F.relu(self.bn2(self.pred_1(feat_4 + fine_3)))
        output = self.pred_2(output)
        return output


class DRN(nn.Module):
    def __init__(self, inplane_albedo, inplane_normal, inplane_lighting):
        super(DRN, self).__init__()
        self.albedoNetwork = basicNetwork(inplane_albedo, 3)
        self.normalNetwork = basicNetwork(inplane_normal, 3)
        self.lightingNetwork = basicNetwork(inplane_lighting, 27)
    
    def forward(self, in_albedo, in_normal, in_lighting):
        albedo = self.albedoNetwork(in_albedo)
        normal = self.normalNetwork(in_normal)
        lighting = self.lightingNetwork(in_lighting)
        return albedo, normal, lighting

if __name__ == '__main__':
		net = HourglassNet()
		x = Variable(torch.Tensor(1, 3, 224, 224))
		albedo, norm, lighting, shading = net(x)
		g = make_dot(shading)
		g.render('HourglassNet')
