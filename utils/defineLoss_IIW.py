import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# senital for divided by 0
eps = 1e-10

class WhdrHingeLoss(nn.Module):
    def __init__(self, margin=0.05, delta=0.1):
        super(WhdrHingeLoss, self).__init__()
        self.margin = margin
        self.delta = delta
    
    def forward(self, v_input, comparisons, numComparisons):
        # record loss
        total_loss = Variable(torch.cuda.FloatTensor([0]))
        numData = v_input.size()[0]
        for count in range(numData):
            loss = Variable(torch.cuda.FloatTensor([0]))
            total_weight = 0
            comparison = comparisons[count]
            for i in range(numComparisons[count]):
                item = list(comparison[i].data.numpy())
                x1, y1, x2, y2, darker = item[0:5]
                weight = float(item[5])
                total_weight += weight
                
                r_1 = v_input[count ,0, y1, x1]
                r_2 = v_input[count ,0, y2, x2]
                
                ratio = r_1 / (r_2 + eps)
                if darker  == 1: 
                    # r_1 is darker than r_2
                    border = 1/(1 + self.delta + self.margin)
                    if (ratio  > border).data[0]:
                    		loss = loss + weight*(ratio - border)
                elif darker == 2:
                    # r_2 is darker than r_1
                    border = 1 + self.delta + self.margin
                    if (ratio < border).data[0]:
                    		loss = loss + weight*(border - ratio)
                elif darker == 0:
                    # r_1 and r_2 are more or less the same
                    if self.margin <= self.delta:
                        border_right = 1 + self.delta - self.margin
                        # loss = max(0, border_left - y, y - border_right)
                        if (ratio > border_right).data[0]:
                        		loss = loss + weight*(ratio - border_right)
                        else:
                        		border_left = 1/border_right
                        		if (ratio < border_left).data[0]:
                        				loss = loss + weight*(border_left - ratio)
                    else:
                        border = 1 + self.delta - self.margin
                        loss = max(1/border-y, y-border)
                else:
                		raise Exception('darker is neighter E(0), 1 or 2')
            loss = loss/total_weight
            total_loss += loss
        
        return total_loss/numData

class WhdrTestLoss(nn.Module):
    def __init__(self, delta=0.1):
        super(WhdrTestLoss, self).__init__()
        self.delta = delta
    
    def forward(self, v_input, comparisons, numComparisons):
        # record loss
        total_loss = Variable(torch.cuda.FloatTensor([0]))
        numData = v_input.size()[0]
        for count in range(numData):
            loss = Variable(torch.cuda.FloatTensor([0]))
            total_weight = 0
            #print comparisons[count]
            comparison = comparisons[count]
            for i in range(numComparisons[count]):
                item = list(comparison[i].data.numpy())
                x1, y1, x2, y2, darker = item[0:5]
                weight = float(item[5])
                total_weight += weight
                
                r_1 = v_input[count ,0, y1, x1]
                r_2 = v_input[count ,0, y2, x2]
                
                ratio = r_1 / (r_2 + eps)
                if darker  == 1: 
                    # r_1 is darker than r_2
                    border = 1/(1 + self.delta)
                    if (ratio  > border).data[0]:
                    		loss = loss + weight*(ratio - border)
                elif darker == 2:
                    # r_2 is darker than r_1
                    border = 1 + self.delta 
                    if (ratio < border).data[0]:
                    		loss = loss + weight*(border - ratio)
                elif darker == 0:
                    # r_1 and r_2 are more or less the same
                    border_right = 1 + self.delta
                    # loss = max(0, border_left - y, y - border_right)
                    if (ratio > border_right).data[0]:
                        loss = loss + weight*(ratio - border_right)
                    else:
                        border_left = 1/border_right
                        if (ratio < border_left).data[0]:
                            loss = loss + weight*(border_left - ratio)
                else:
                	raise Exception('darker is neighter E(0), 1 or 2')
            loss = loss/total_weight
            total_loss += loss
        
        return total_loss/numData

class WhdrTestLoss_Paper(nn.Module):
    def __init__(self, delta=0.1):
        super(WhdrTestLoss_Paper, self).__init__()
        self.delta = delta
    
    def forward(self, v_input, comparisons, numComparisons):
        # record loss
        total_loss = Variable(torch.cuda.FloatTensor([0]))
        numData = v_input.size()[0]
        for count in range(numData):
            loss = Variable(torch.cuda.FloatTensor([0]))
            total_weight = 0
            #print comparisons[count]
            comparison = comparisons[count]
            for i in range(numComparisons[count]):
                item = list(comparison[i].data.numpy())
                x1, y1, x2, y2, darker = item[0:5]
                weight = float(item[5])
                total_weight += weight
                
                r_1 = v_input[count ,0, y1, x1]
                r_2 = v_input[count ,0, y2, x2]
                if (r_2/(r_1 + eps) > 1.0 + self.delta).data[0]:
                    # darker
                    alg = 1
                elif (r_1/(r_2 + eps) > 1.0 + self.delta).data[0]:
                    alg = 2
                else:
                    alg = 0
                if alg != darker:
                    loss += weight
            
            loss = loss/total_weight
            total_loss += loss
        return total_loss/numData
