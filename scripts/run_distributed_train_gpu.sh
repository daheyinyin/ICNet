#!/bin/bash

# ============================================================================

if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES] [PROJECT_PATH]"
    echo "Please run the script as: "
    echo "bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES] [PROJECT_PATH]"
    echo "for example: bash scripts/run_distributed_train_gpu.sh 8 0,1,2,3,4,5,6,7 /root/ICNet/"
    echo "=============================================================================================================="
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
    echo "error: PROJECT_PATH=$PATH3 is not a directory"
exit 1
fi

export RANK_SIZE=$1
export DEVICE_NUM=$1
export CUDA_VISIBLE_DEVICES=$2

rm -rf train_dis
mkdir ./train_dis
cp ./*.py ./train_dis
cp -r ./src ./train_dis
cd ./train_dis || exit

env > env.log

echo "start training in $DEVICE_NUM device."

mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
python3 train.py --project_path=$PATH3 --run_distribute=True --device_target='GPU'  > log.txt 2>&1 &

cd ..
