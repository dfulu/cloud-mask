import numpy as np
import pandas as pd
import torch

import argparse
import time
from tqdm import tqdm
from pathlib import Path

from datadrivencloud.data import tiffImageDataset
from datadrivencloud.modules import CloudModel

from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump

############################################
# setup

t0 = time.time()
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--checkpoint', type=str, nargs='+')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--fraction', type=float, default=0.1)
args = parser.parse_args()

np.random.seed(args.seed)

device = torch.device("cpu" if args.gpu==-1 else f"cuda:{args.gpu}")

models = [CloudModel.load_from_checkpoint(path).eval().to(device) for path in args.checkpoint]

def apply_models_to_cpu(x):
    y = []
    for model in models:
        y += [model(x).detach().cpu()]
    return torch.cat(y, dim=1) # cat along channel axis

################################################################################
# Construct dataset to train on

df = pd.read_csv("../data/train_metadata_filtered.csv")

dataloader = torch.utils.data.DataLoader(
    tiffImageDataset(df, root_path="/home/s1205782/Datastore/Projects/cloudmask/data/train_{}"),
    shuffle=False,
    batch_size=6,
    num_workers=4
)


X_train = []
y_train = []

for batch in tqdm(dataloader):
    images, labels = batch
    preds = apply_models_to_cpu(images.to(device)).numpy()
    
    preds = np.transpose(preds, axes=(0,2,3,1)) # move channels to last
    labels = np.transpose(labels.numpy(), axes=(0,2,3,1))
    
    preds = preds.reshape(-1, preds.shape[-1])
    labels = labels.reshape(-1, labels.shape[-1])
    
    mask = np.random.choice([True, False], preds.shape[0],  p=[args.fraction, 1-args.fraction])
    
    X_train += [preds[mask]]
    y_train += [labels[mask]]


X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

X_train, y_train = shuffle(X_train, y_train)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
dump(model, 'meta_learner.joblib')

print(f"took {(time.time() - t0)/60.} minutes")