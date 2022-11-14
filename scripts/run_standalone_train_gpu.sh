
if [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [PROJECT_PATH] [DEVICE_ID]"
    echo "for example: bash scripts/run_standalone_train_gpu.sh /root/ICNet/ 0"
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

PATH1=$(get_real_path $1)



if [ ! -d $PATH1 ]
then
    echo "error: PROJECT_PATH=$PATH1 is not a directory"
exit 1
fi

export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$2
echo "start training for device $DEVICE_ID"
env > env.log

if [ -d "train" ]; then
    rm -rf ./train
fi
mkdir ./train
cp ./*.py ./train
cp -r ./src ./train
cd ./train || exit


python3 train.py --project_path=$PATH1 --device_target='GPU' --device_id=$DEVICE_ID > log.txt 2>&1 &

cd ..
