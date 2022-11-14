#!/bin/bash

# ============================================================================

if [ $# != 2 ]
then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]"
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


if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi


export SERVER_ID=0
ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
rank_start=$((DEVICE_NUM * SERVER_ID))
first_device=0
export RANK_TABLE_FILE=$PATH1

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$((first_device+i))
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --data_url=$PATH2 --run_distribute=True --device_id=$DEVICE_ID  &> log &

    cd ..
done
