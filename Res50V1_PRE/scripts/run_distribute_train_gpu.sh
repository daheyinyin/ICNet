#!/bin/bash

# ============================================================================

if [ $# != 3 ]
then
  echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES] [DATASET_PATH]"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}


PATH3=$(get_real_path $3)



if [ ! -d $PATH3 ]
then
    echo "error: DATASET_PATH=$PATH3 is not a directory"
exit 1
fi


export RANK_SIZE=$1
export DEVICE_NUM=$1
export CUDA_VISIBLE_DEVICES=$2


rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit
echo "start training on $2"
env > env.log
nohup mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
python train.py --run_distribute True --device_target GPU --data_url=$PATH3 &> log &

