#!/bin/sh
runname=$1
flags=$2
echo $runname
echo $flags

rm -rf baselines/ood/output/${runname}
mkdir baselines/ood/output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

corp=sri
screen -dmS ${runname}_${corp} 
screen -S ${runname}_${corp} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
screen -S ${runname}_${corp} -X stuff "CUDA_VISIBLE_DEVICES=0 python baselines/ood/train_ood.py --ckpt_name ${runname}/${corp} ${flags} --corpus ${corp}^M"

corp=lb
screen -dmS ${runname}_${corp} 
screen -S ${runname}_${corp} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
screen -S ${runname}_${corp} -X stuff "CUDA_VISIBLE_DEVICES=1 python baselines/ood/train_ood.py --ckpt_name ${runname}/${corp} ${flags} --corpus ${corp}^M"