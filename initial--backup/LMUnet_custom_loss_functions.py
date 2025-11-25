import torch
import torch.nn as nn

class HuberLossNan(nn.HuberLoss):
    """ Adjusted Huber loss class masking nan values."""
    
    def __init__(self, reduction='mean', delta=1.0):
        super(HuberLossNan, self).__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):        
        # Create a nan mask for all nan values in both input and target
        nan_mask = torch.isnan(input) | torch.isnan(target)
        valid_mask = ~nan_mask
        
        # Apply the mask to both input and target to keep same shape
        input_valid = input[valid_mask]
        target_valid = target[valid_mask]
        
        # Calculate the loss for masked input and target
        loss = super(HuberLossNan, self).forward(input_valid, target_valid)
        return loss

class MSELossNan(nn.MSELoss):
    """ Adjusted MSE loss class masking nan values."""
    
    def __init__(self, reduction='mean', delta=1.0):
        super(MSELossNan, self).__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):
        # Create a nan mask for all nan values in both input and target
        nan_mask = torch.isnan(input) | torch.isnan(target)
        valid_mask = ~nan_mask

        # Apply the mask to both input and target to keep same shape
        input_valid = input[valid_mask]
        target_valid = target[valid_mask]
        
        # Calculate the loss for masked input and target
        loss = super(MSELossNan, self).forward(input_valid, target_valid)
        return loss