#!/bin/sh
runname=$1
flags=$2
echo $runname
echo $flags

rm -rf output/${runname}
mkdir output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

env="dev"
screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=0 python run.py --run_name ${runname}/${env} --env_name ${env} --feat_root /mnt/usb1/AfricanContinentLID/wavlm/,/mnt/usb1/AfricanContinentLID/xvectors/ --context 0 ${flags}; exec bash"

env="test"
screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=1 python run.py --run_name ${runname}/${env} --env_name ${env} --feat_root /mnt/usb1/AfricanContinentLID/wavlm/,/mnt/usb1/AfricanContinentLID/xvectors/ --context 0 ${flags}; exec bash"