import numpy as np
import pandas as pd
import torch

import argparse
import time
from tqdm import tqdm
from pathlib import Path

from datadrivencloud.data import tiffImageDataset
from datadrivencloud.modules import CloudModel

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
        y = y + model(x).detach().cpu()
    y = y/len(models)
    return y

if args.clean:
    df = pd.read_csv("../data/train_metadata_filtered.csv")
else:
    df = pd.read_csv("../data/train_metadata.csv")

dataloader = torch.utils.data.DataLoader(
    tiffImageDataset(df),
    shuffle=False,
    batch_size=4,
    num_workers=4
)


############################################
# functions

def confusion_values(pred, y):
    TP = torch.logical_and(pred, y).sum().item()
    TN = torch.logical_and(~pred, ~y).sum().item()
    FP = torch.logical_and(pred, ~y).sum().item()
    FN = torch.logical_and(~pred, y).sum().item()
    return TP, TN, FP, FN

############################################
# main

# these thresholds chosenj based on work in notebook 005
thresholds = np.linspace(0.1, 0.65)
df_thresh = pd.DataFrame(
    {k:np.zeros(thresholds.shape[0], dtype=float) for k in ['TP', 'TN', 'FP', 'FN']},
    index=thresholds
)

pixels_per_image = 512**2

for batch in tqdm(dataloader):
    images, labels = batch
    preds = apply_model(images.to(device)).cpu()
    for thresh in thresholds:
        values = confusion_values(preds>thresh, labels.bool())
        df_thresh.loc[thresh] += np.array([*values], dtype=float)/pixels_per_image

df_thresh = df_thresh/df.shape[0]

df_thresh['IoU'] = df_thresh['TP']/df_thresh[['TP', 'FP', 'FN']].sum(axis=1)

if args.clean:
    filename = f"threshold_stats_clean.csv"
else:
    filename = f"threshold_stats_all.csv"

savepath = f"{args.outdir}/{filename}"
df_thresh.to_csv(savepath)
print(f"Threshold stats saved to {savepath}")
print(f"took {(time.time() - t0)/60.} minutes")