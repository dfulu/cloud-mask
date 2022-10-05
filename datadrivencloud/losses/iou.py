import torch 

def iou_metric(pred, y, p=0.5):
    pred_bin = pred>p
    y_bin = y.bool()
    intersection = torch.logical_and(pred_bin, y_bin)
    union = torch.logical_or(pred_bin, y_bin)
    return intersection.sum()/union.sum()
    
    