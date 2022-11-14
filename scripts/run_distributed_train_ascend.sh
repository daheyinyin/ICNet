#!/bin/bash

# ============================================================================

if [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_distributed_train_ascend.sh [RANK_TABLE_FILE] [OUTPUT_PATH]"
    echo "Please run the script as: "
    echo "bash scripts/run_distributed_train_ascend.sh [RANK_TABLE_FILE] [OUTPUT_PATH]"
    echo "for example: bash scripts/run_distributed_train_ascend.sh /absolute/path/to/RANK_TABLE_FILE /root/ICNet/"
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

PATH2=$(get_real_path $2)


if [ ! -d $PATH2 ]
then
    echo "error: PROJECT_PATH=$PATH2 is not a directory"
exit 1
fi

export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=8

for((i=0;i<$RANK_SIZE;i++))
do
    rm -rf train_dis$i
    mkdir ./train_dis$i
    cp ./*.py ./train_dis$i
    cp -r ./src ./train_dis$i
    cd ./train_dis$i || exit
    export RANK_TABLE_FILE=$1
    export RANK_SIZE=8
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log

    python3 train.py --project_path=$PATH2 --run_distribute=True --device_target='Ascend' --device_id=$DEVICE_ID  > log.txt 2>&1 &

    cd ../
done
