from torch import nn
import torch

class Div2000(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image: torch.Tensor):
        return image / 2000
    
    def inverse(self, image: torch.Tensor):
        return image * 2000
    

class LogTransform(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image: torch.Tensor):
        # These rough numbers derived from notebook 003
        return (torch.log(image+1) - 6) / 2
    
    def inverse(self, image: torch.Tensor):
        return torch.exp((image*2) + 6) - 1


class QuarterPower(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image: torch.Tensor):
        # These rough numbers derived from notebook 003
        return ((image**0.25) - 6) / 4
    
    def inverse(self, image: torch.Tensor):
        return ((image*4) + 6)**4
    
class Visual(nn.Module):
    def __init__(self, pixel_max=1, c=10.0, th=0.125):
        super().__init__()
        self.pixel_max = pixel_max
        self.c = c
        self.th = th
    
    def forward(self, image: torch.Tensor):
        # These rough numbers derived from notebook 003
        max_val = image.amax(dim=(1,2,3), keepdim=True)
        min_val = image.amin(dim=(1,2,3), keepdim=True)
        range_val = max_val - min_val
        norm = (image - min_val) / range_val
        norm = 1 / (1 + torch.exp(self.c * (self.th - norm)))
        return norm * self.pixel_max
