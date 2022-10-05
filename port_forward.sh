pip install pytorch-lightning
pip install segmentation-models-pytorch
pip install kornia

cd cloudmask
pip install -e .

nohup tensorboard --logdir lightning_logs --bind_all --port 6006 &

ssh -R 6006:localhost:6006 s1205782@ssh.geos.ed.ac.uk