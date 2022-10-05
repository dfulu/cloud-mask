"""
Strips all the data fromthe checkpoint needed by pytorch lightning, and therefore
reduces checkpoint file size for upload and inference.

"""

from model_modules import CloudModel
from apply_model import TransformModel
import torch
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('checkpoints', type=str, nargs='+',
                    help='checkpoints to strip')
parser.add_argument('--outdir', type=str, default='.')
args = parser.parse_args()

for i, ckpt in enumerate(tqdm(args.checkpoints)):
    new_name = f"checkpoint_{i}.ckpt"
    print(f"{new_name} : {ckpt}")
    lightning_checkpoint = CloudModel.load_from_checkpoint(ckpt).eval()
    new_model = TransformModel(
        lightning_checkpoint.model, 
        lightning_checkpoint.transform,
        lightning_checkpoint.bands,
    )
    torch.save(new_model, f"{args.outdir}/{new_name}")