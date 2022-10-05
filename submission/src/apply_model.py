#from modules import CloudModel
import numpy as np
from scipy.stats import trim_mean as _trim_mean
import torch
import joblib
from itertools import cycle


class TransformModel(torch.nn.Module):
    
    def __init__(self, model, transform, bands):
        super().__init__()
        self.model = model
        self.transform = transform
        self.bands = bands
        
    def forward(self, image: torch.Tensor):
        return self.model(self.transform(image[:, self.bands]))

class FlipAugmentation(torch.nn.Module):
    
    def __init__(self, vertical=True, horizontal=True):
        super().__init__()
        self.vertical = vertical
        self.horizontal = horizontal
        
    def forward(self, image: torch.Tensor):
        x = image
        if self.vertical:
            x = torch.flip(x, [-1])
        if self.horizontal:
            x = torch.flip(x, [-2])
        return x
    
    def inverse(self, image: torch.Tensor):
        return self.forward(image)
    
def trim_mean(trim):
    def f(x, axis=None):
        x = np.array(x)
        if trim>=1:
            t = trim/float(x.shape[axis]-0.5)
        else:
            t = trim
        return _trim_mean(x, t, axis=axis)
    return f

def apply_meta_learner(path):
    model = joblib.load(path)
    def f(x, *args, **kwargs):
        preds = np.array(x)
        target_shape = preds.shape[1:]
        preds = preds.reshape(preds.shape[0], -1).T
        return model.predict(preds).T.reshape(target_shape)
    return f

def model_function(ckpt_paths, reduction, thresh=0.5, augmentations=0, gpu=0):
    device = torch.device("cpu" if gpu==-1 else f"cuda:{gpu}")
    if augmentations:
        aug_funcs = [
            FlipAugmentation(False, False),
            FlipAugmentation(False, True),
            FlipAugmentation(True, False),
            FlipAugmentation(True, True), 
        ]
        if augmentations==1:
            aug_funcs = aug_funcs
        else:
            aug_funcs = aug_funcs[:augmentations]
        aug_funcs = cycle(aug_funcs)
        
    models = [torch.load(path).eval().to(device) for path in ckpt_paths]
    
    if '.joblib' in reduction:
        reduce = apply_meta_learner(reduction)
    elif 'trim_mean' in reduction:
        reduce = trim_mean(float(reduction.split('-')[-1]))
    else:
        reduce = {
            'mean':np.mean,
            'min':np.min,
            'max':np.max,
            'median':np.median,
        }[reduction]
    
    def apply_model(x):
        x = x.to(device)
        if augmentations:
            y = []
            for model in models:
                this_y = 0
                for i in range(augmentations):
                    aug = next(aug_funcs)
                    this_y = this_y + aug.inverse(model(aug.forward(x))).detach().cpu().numpy()
                y.append(this_y/augmentations)
        else:
            y = [model(x).detach().cpu().numpy() for model in models]
        y = reduce(y, axis=0)
        y = y > thresh
        return y.astype("uint8")
    return apply_model