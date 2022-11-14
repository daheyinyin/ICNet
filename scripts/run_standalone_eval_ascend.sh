#!/bin/bash

if [ $# != 4 ]
then
    echo "Usage: bash scripts/run_standalone_eval_ascend.sh [DATASET_PATH] [CHECKPOINT_PATH] [PROJECT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
PATH3=$(get_real_path $3)


if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: PROJECT_PATH=$PATH3 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$4
export RANK_SIZE=1
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ./eval.py ./eval
cp -r ./src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"
python eval.py --dataset_path=$PATH1 --checkpoint_path=$PATH2 --project_path=$PATH3 --device_id=$DEVICE_ID &> log &

cd ..
