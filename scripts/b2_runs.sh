# bash b1_runs.sh [0|1|2]

function run {
  echo $2
  python train.py --seed $2 --loss_name symlovasz --backbone efficientnet-b2 --lr 1e-3 --run_name ens-b2-gpu$1 --gpu $1
}


for i in 0 1 2
do
  n=$(( $1 + $i*3 ))
  echo $n
  seed=$(( $n  + 26012022 ))
  echo $seed
  run $1 $seed
  mv ../lightning_logs/ens-b2-gpu$1/version_0 ../lightning_logs/ens-b2/version_$n
  rm -rf ../lightning_logs/ens-b2-gpu$1
done


