"""
see
https://arxiv.org/pdf/2006.14822.pdf

"""

import torch
from torch import nn
import torch.nn.functional as F
from . lovasz_losses import lovasz_hinge

class Reverse(torch.nn.Module):
    def __init__(self, lossobject):
        super(Reverse, self).__init__()
        self.lossobject = lossobject
        
    def forward(self, inputs, targets):
        return self.lossobject(1-inputs, 1-targets)
    
class Symmetric(torch.nn.Module):
    def __init__(self, lossobject):
        super(Symmetric, self).__init__()
        self.lossobject = lossobject
        
    def forward(self, inputs, targets):
        return 0.5*(self.lossobject(1-inputs, 1-targets) + self.lossobject(inputs, targets))
        

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=1, eps=1e-4):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps=eps

    def forward(self, inputs, targets):
        # CCE
        cross_ent =  F.binary_cross_entropy(inputs, targets)

        # RCE
        # inputs = torch.clamp(inputs, min=self.eps, max=1.0-self.eps)
        targets = torch.clamp(targets, min=self.eps, max=1.0-self.eps)
        reverse_cross_ent =  F.binary_cross_entropy(targets, inputs)

        # Loss
        loss = self.alpha * cross_ent.mean() + self.beta * reverse_cross_ent.mean()
        return loss
    
#https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
class DiceLoss(nn.Module):
    def __init__(self, eps=1):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.eps)/(inputs.sum() + targets.sum() + self.eps)  
        
        return 1 - dice
    
    
class DiceBCELoss(nn.Module):
    def __init__(self, eps=1, alpha=0.5):
        super(DiceBCELoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.dice_loss = DiceLoss(eps)

    def forward(self, inputs, targets):
        
        dice_loss = self.dice_loss(inputs, targets)    
    
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.alpha*BCE + (1-self.alpha)*dice_loss
        
        return Dice_BCE
    

class DiceSCELoss(nn.Module):
    def __init__(self, k_dice=1, k_sce=0.07, eps=1):
        super(DiceSCELoss, self).__init__()
        self.eps = eps
        self.dice_loss = DiceLoss(eps)
        self.sce_loss = SCELoss()
        self.k_dice = k_dice
        self.k_sce = k_sce

    def forward(self, inputs, targets):
        
        dice = self.dice_loss(inputs, targets)    
    
        sce = self.sce_loss(inputs, targets)
        dice_sce = self.k_sce*sce + self.k_dice*dice
        
        return dice_sce
    

class IoULoss(nn.Module):
    def __init__(self, eps=1):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.eps)/(union + self.eps)
                
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1-bce_exp)**self.gamma * bce
                       
        return focal_loss
    
    
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                
        return Lovasz