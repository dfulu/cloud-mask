import numpy as np
from apply_model import model_function
import yaml
import pandas as pd
from data import tiffImageDataset
import argparse
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image
from glob import glob
from datetime import datetime
import torch
import dask

t0 = time.time()
print(datetime.now())

###########################################
# inputs 

# load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
with open('reduction.yaml', 'r') as file:
    config.update(yaml.safe_load(file))

with open('submission_summary.txt', 'r') as file:
    print(''.join(file.readlines()))
    
input_dir_root = f"{config['root_directory']}/data/test_"+"{}"
output_dir = f"{config['root_directory']}/predictions"

batch_size = config['batch_size']
num_workers = config['num_workers']

############################################
# functions

def calc_iou(pred, y, eps=1):
    """IoU for my local testing only"""
    pred = pred.astype(bool)
    y = y.astype(bool)
    intersection = np.logical_and(pred, y)
    union = np.logical_or(pred, y)
    return intersection.sum()/(union.sum()+eps)

def calculate_confusion_values(pred, y):
    TP = np.logical_and(pred, y).sum().item()
    TN = np.logical_and(~pred, ~y).sum().item()
    FP = np.logical_and(pred, ~y).sum().item()
    FN = np.logical_and(~pred, y).sum().item()
    return TP, TN, FP, FN

    
############################################
# setup



# construct dataset and return labels if testing locally
df = pd.read_csv(f"{config['root_directory']}/data/test_metadata.csv")
dataset = tiffImageDataset(df, root_path=input_dir_root, return_label=config['local_test'])
dataloader = torch.utils.data.DataLoader(
    tiffImageDataset(
        df, 
        root_path=input_dir_root, 
        return_label=config['local_test']
    ),
    shuffle=False,
    batch_size=batch_size,
    num_workers=num_workers,
)

apply_model_function = model_function(
    glob('checkpoints/checkpoint*.ckpt'), 
    config['reduction'], 
    config['threshold'],
    config['augmentation'],
    config['gpu'],
)

if config['local_test']:
    xs = []
    ys = []
    preds =[]
    ious = []
    s = ''
    
# store stats to help debug
band_mins = np.full(4, np.inf)
band_maxs = np.full(4, 0)
band_means = 0
band_mean_squares = 0
positive_predictions = 0
all_positive_preds = []
N = len(df)
pixels_per_image = 512**2
n_pixels = N * pixels_per_image

    
############################################
# predict
@dask.delayed
def save(image, prediction, chip_id):         
    output_path = f"{output_dir}/{chip_id}.tif"
    im = Image.fromarray(prediction[0])
    im.save(output_path)

miniters = 1 if config['local_test'] else max(1, int(len(df)/batch_size/100))

chip_n = 0
for batch in tqdm(dataloader,  miniters=miniters):

    if config['local_test']:
        images, labels = batch
    else:
        images = batch

    predictions = apply_model_function(images)
    
    #dask.compute([save(images[i], predictions[i], df.iloc[chip_n+i].chip_id) for i in range(len(images))], scheduler='threads')
    
    for i in range(len(images)):
        image = images[i]
        prediction = predictions[i]
        chip_id = df.iloc[chip_n+i].chip_id

        output_path = f"{output_dir}/{chip_id}.tif"
        im = Image.fromarray(prediction[0])
        im.save(output_path)
        # store some things
        all_positive_preds += [prediction.astype(float).sum()/pixels_per_image]
        band_means += image.sum(dim=(1,2)).numpy()/n_pixels
        band_mean_squares += (image**2).sum(dim=(1,2)).numpy()/n_pixels
        image_cpu = image.numpy()
        band_maxs = np.stack([band_maxs, image_cpu.max(axis=(1,2))]).max(axis=0)
        band_mins = np.stack([band_mins, image_cpu.min(axis=(1,2))]).min(axis=0)
        if config['local_test']:
            label = labels[i].numpy()
            iamge = image.numpy()
            xs += [iamge]
            ys += [label]
            preds += [prediction]
            ious += [calc_iou(prediction, label)]
            s += f"{chip_id}: IoU = {ious[-1]}\n"
    chip_n += len(images)
        
    
del dataloader, dataset
############################################
# local testing

if config['local_test']:
    print(s)
    print(f"mean IoU = {np.mean(ious)}")

    from datadrivencloud.loggers import create_image_figure
    import matplotlib.pyplot as plt
    
    xs = torch.from_numpy(np.stack(xs).astype(np.float32)).float()
    ys = torch.from_numpy(np.stack(ys).astype(np.float32)).float()
    preds = torch.from_numpy(np.stack(preds).astype(np.float32)).float()
    fig = create_image_figure(xs, preds, ys, n=xs.shape[0], random_image=False)
    plt.savefig('../local_test_results.png')
    
    if config['test_all']:

        df = pd.read_csv("/home/s1205782/Datastore/Projects/cloudmask/data/train_metadata_filtered.csv")
        

        dataloader = torch.utils.data.DataLoader(
            tiffImageDataset(df, root_path="/home/s1205782/Datastore/Projects/cloudmask/data/train_{}"),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        confusion_values = 0
        for batch in tqdm(dataloader):
            images, labels = batch
            preds = apply_model_function(images)
            cv = calculate_confusion_values(preds.astype(bool), labels.numpy().astype(bool))
            confusion_values += np.array([*cv], dtype=float)/pixels_per_image

        confusion_values = confusion_values/df.shape[0]
        all_iou = confusion_values[0]/confusion_values[[0,2,3]].sum()
        confusion_dict = {k:v for k, v in zip(["TP", "TN", "FP", "FN"], confusion_values)}
        confusion_dict['IoU'] = all_iou
        print(confusion_dict)
        
################################################################################
# finally
if config['local_test']:
    print(f"# of datapoints : {N}")
    print(f"Positive prediction fraction : {np.mean(all_positive_preds)}")
    print(f"band means : {band_means.astype(int)}")
    print(f"band stds : {((band_mean_squares - band_means**2)**0.5).astype(int)}")
    print(f"band mins : {band_mins.astype(int)}")
    print(f"band maxes : {band_maxs.astype(int)}")
print(f"Time taken : {(time.time()-t0)/60:.1f} mins")


    