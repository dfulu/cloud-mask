# done
# python train.py --seed 45544     --loss_name dicebce  --backbone efficientnet-b0 # different seed but oh well
# python train.py --seed 234578632 --loss_name lovasz   --backbone efficientnet-b0 --lr 1e-4  --run_name base-lovasz
# python train.py --seed 234578632 --loss_name dice     --backbone efficientnet-b0 --lr 2e-4  --run_name base-dice
# python train.py --seed 234578632 --loss_name bce      --backbone efficientnet-b0 --lr 1e-4  --run_name base-bce
# python train.py --seed 234578632 --loss_name sce      --backbone efficientnet-b0 --lr 1e-4  --run_name base-sce
# python train.py --seed 234578632 --loss_name dicebce  --backbone efficientnet-b0            --run_name base-seed
# python train.py --seed 45544     --loss_name dicebce  --backbone efficientnet-b0            --run_name base-seed
# python train.py --seed 234578632 --loss_name dicebce  --backbone efficientnet-b1            --run_name base-eff-b1
# python train.py --seed 234578632 --loss_name dicebce  --backbone efficientnet-b0 --uplus    --run_name base-unetpp
# python train.py --seed 234578632 --loss_name bce  --backbone efficientnet-b0 --robust    --run_name base-robustbce

# to do
python train.py --seed 45544     --loss_name dicesce? --backbone efficientnet-b0 --lr 1e-4  --run_name base-dicesce

# later on ensemble using different seeds and different models from above

# -------------- ensemble 0  ----------------------------

# done
# python train.py --seed 234578632 --loss_name dicesce  --backbone efficientnet-b1 --lr 2e-4   --run_name ens-0

# -------------- ensemble 1  ----------------------------

# done
# python train.py --seed 123242443 --loss_name dicebce  --backbone efficientnet-b1 --lr 2e-4   --run_name ens-1
# python train.py --seed 47484833 --loss_name dicebce  --backbone efficientnet-b1 --lr 1e-3   --run_name ens-1
# python train.py --seed 45849589 --loss_name dicebce  --backbone efficientnet-b1 --lr 1e-3   --run_name ens-1
# python train.py --seed 8459595 --loss_name dicebce  --backbone efficientnet-b1 --lr 1e-3   --run_name ens-1
# python train.py --seed 24564357 --loss_name dicebce  --backbone efficientnet-b1 --lr 1e-3   --run_name ens-1
# python train.py --seed 97834762 --loss_name dicebce  --backbone efficientnet-b1 --lr 1e-3   --run_name ens-1
# python train.py --seed 34729379 --loss_name dicebce  --backbone efficientnet-b1 --lr 1e-3   --run_name ens-1

python train.py --seed 778565 --loss_name dicebce --backbone efficientnet-b1 --lr 1e-3 --run_name ens-3 --gpu 2
python train.py --seed 324155 --loss_name dicebce --backbone efficientnet-b1 --lr 1e-3 --run_name ens-3 --gpu 1
python train.py --seed 155641 --loss_name dicebce --backbone efficientnet-b1 --lr 1e-3 --run_name ens-3 --gpu 0


# ----------------- test --------------------------------
python train.py --seed 646568958 --loss_name dicebce  --backbone efficientnet-b0 --lr 1e-3   --run_name test --dev --cpu




python train.py --seed 5324356 --loss_name dicebce --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 0
python train.py --seed 5324356 --loss_name lovasz --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 1
python train.py --seed 5324356 --loss_name symlovasz --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 2 # run before symmetric fix

python train.py --seed 5324356 --loss_name symdicebce --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 0 # run after symmetric fix
python train.py --seed 5324356 --loss_name dicebce50 --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 1
python train.py --seed 5324356 --loss_name dicesce --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 2

python train.py --seed 5324356 --loss_name dice --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 0
python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 1
python train.py --seed 5324356 --loss_name dicebce --backbone efficientnet-b1 --lr 1e-3 --preproc Div2000 --run_name exp-3 --gpu 2

python train.py --seed 5324356 --loss_name jaccard --backbone resnet34 --lr 1e-3 --run_name exp-3 --gpu 0
python train.py --seed 5324356 --loss_name jaccard --backbone resnet50 --lr 1e-3 --run_name exp-3 --gpu 1
python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b2 --lr 1e-3 --run_name exp-3 --gpu 2

python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b0 --lr 1e-3 --run_name exp-3 --gpu 0
python train.py --seed 5324356 --loss_name jaccard --backbone resnet152 --lr 1e-3 --run_name exp-3 --gpu 1
python train.py --seed 5324356 --loss_name jaccard --backbone densenet201 --lr 1e-3 --run_name exp-3 --gpu 2

python train.py --seed 5324356 --loss_name jaccard --backbone inceptionv4 --lr 1e-3 --run_name exp-3 --gpu 0
python train.py --seed 5324356 --loss_name dicebce --backbone efficientnet-b1 --lr 1e-3 --preproc Div2000 --run_name exp-3 --gpu 1
python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b3 --lr 1e-3 --run_name exp-3 --gpu 2

python train.py --seed 5324356 --loss_name jaccard --backbone densenet201 --lr 3e-3 --run_name exp-3 --gpu 2
python train.py --seed 5324356 --loss_name jaccard --backbone densenet201 --lr 3e-4 --run_name exp-3 --gpu 2

python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --preproc Div2000 --run_name exp-3 --gpu 0

python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --preproc Visual --run_name exp-3 --gpu 0
python train.py --seed 5324356 --loss_name jaccard --backbone efficientnet-b1 --lr 3e-4 --run_name exp-3 --gpu 2

python train.py --seed 6658961 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 1

python train.py --seed 6658961 --loss_name jaccard --backbone efficientnet-b3 --lr 1e-3 --run_name exp-3 --gpu 0
python train.py --seed 457856 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name exp-3 --gpu 1
python train.py --seed 457856 --loss_name jaccard --backbone efficientnet-b3 --lr 1e-3 --run_name exp-3 --gpu 2

# you are here
cd Datastore/Projects/cloudmask/scripts
conda activate cloud2

mv ../lightning_logs/ens-4/version_0 ../lightning_logs/exp-3/version_30




# increase number of epochs first
python train.py --seed 6658961 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 0
python train.py --seed 4544412 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 1
python train.py --seed 2564832 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 2

python train.py --seed 1256687 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 0
python train.py --seed 7863434 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 1
python train.py --seed 2135512 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-4 --gpu 2