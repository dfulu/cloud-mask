import numpy as np
import pandas as pd
import torch

import argparse
import time
from tqdm import tqdm
from pathlib import Path

from datadrivencloud.data import tiffImageDataset
from datadrivencloud.modules import CloudModel
from skimage.segmentation import random_walker


############################################
# setup

t0 = time.time()
parser = argparse.ArgumentParser(description='')

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--clean', action='store_true')
parser.add_argument('--checkpoint', type=str, nargs='+')
parser.add_argument('--outdir', type=str, default='.')
args = parser.parse_args()

device = 'cpu' if args.cpu else 'cuda'
models = [CloudModel.load_from_checkpoint(path).eval().to(device) for path in args.checkpoint]


def apply_model(x):
    y = 0
    for model in models:
        y = y + model(x).detach().cpu().numpy()
    y = y/len(models)
    return y

if args.clean:
    df = pd.read_csv("/home/jovyan/cloudmask/data/train_metadata_filtered.csv")
else:
    df = pd.read_csv("/driven-data/cloud-cover/train_metadata.csv")
    
if args.clean:
    filename = f"rand_walk_threshold_stats_clean.csv"
else:
    filename = f"rand_walk_threshold_stats_all.csv"
savepath = f"{args.outdir}/{filename}"


dataloader = torch.utils.data.DataLoader(
    tiffImageDataset(df),
    shuffle=True,
    batch_size=8,
    num_workers=4,
)


############################################
# functions

def confusion_values(pred, y):
    TP = np.logical_and(pred, y).sum()
    TN = np.logical_and(~pred, ~y).sum()
    FP = np.logical_and(pred, ~y).sum()
    FN = np.logical_and(~pred, y).sum()
    return TP, TN, FP, FN

def walker(preds, low=0.2, high=0.8):
    labels = np.zeros_like(preds, dtype=int)
    labels[preds<low] = -1
    labels[preds>high] = 1
    for i in range(len(preds)):
        labels[i] = random_walker(preds[i], labels[i])
    labels[labels==0] = -1
    return (labels + 1)/2

############################################
# main

# these thresholds chosenj based on work in notebook 005
thresholds = np.linspace(0.1, 0.5, num=20)
df_thresh = pd.DataFrame(
    {k:np.zeros(thresholds.shape[0], dtype=float) for k in ['TP', 'TN', 'FP', 'FN']},
    index=thresholds
)

pixels_per_image = 512**2

for batch in tqdm(dataloader):
    images, labels = batch
    preds = apply_model(images.to(device))
    for thresh in thresholds:
        values = confusion_values(walker(preds, low=thresh, high=.5).astype(bool), labels.bool())
        df_thresh.loc[thresh] += np.array([*values], dtype=float)/pixels_per_image
    df_thresh.to_csv(savepath)

df_thresh = df_thresh/df.shape[0]

df_thresh['IoU'] = df_thresh['TP']/df_thresh[['TP', 'FP', 'FN']].sum(axis=1)

df_thresh.to_csv(savepath)

print(f"Threshold stats saved to {savepath}")
print(f"took {(time.time() - t0)/60.} minutes")