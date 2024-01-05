#!/bin/sh
runname=$1
device=$2
echo $runname
echo $device

for dd in output/${runname}/*; do
    if [[ $dd == *"rm"* ]]
    then
        mm="train_lb"
    else
        mm="train_sri"
    fi

    if [ -z $device ]
    then
        CUDA_VISIBLE_DEVICES=0 python baselines/ood/eval_ood.py --eval_run ${dd} --ckpt_name ${mm}
    else
        CUDA_VISIBLE_DEVICES=${device} python baselines/ood/eval_ood.py --eval_run ${dd} --ckpt_name ${mm}
    fi
done