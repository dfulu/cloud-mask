# bash b1_runs.sh [0|1|2]

function run {
  echo $2
  python train.py --seed $2 --loss_name jaccard --backbone efficientnet-b1 --lr 1e-3 --run_name ens-b1-gpu$1 --gpu $1
}


for i in 2 3 4 5
do
  n=$(( $1*6 + $i ))
  echo $n
  seed=$(( $n  + 26012022 ))
  echo $seed
  run $1 $seed
  mv ../lightning_logs/ens-b1-gpu$1/version_0 ../lightning_logs/ens-b1/version_$n
  rm -rf ../lightning_logs/ens-b1-gpu$1
done


