#!/bin/sh
runname=$1
flags=$2
echo $runname
echo $flags

rm -rf output/${runname}
mkdir output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

for lang in ha yo bas; do
    screen -dmS ${runname}_${lang} sh -c "CUDA_VISIBLE_DEVICES=0 python run.py --run_name ${runname}/${lang} --env_name test --lid_target ${lang} --feat_root /data1/AfricanContinentLID/ecapalang/ ${flags}; exec bash"
done

for lang in ckb kmr sr; do
    screen -dmS ${runname}_${lang} sh -c "CUDA_VISIBLE_DEVICES=1 python run.py --run_name ${runname}/${lang} --env_name test --lid_target ${lang} --feat_root /data1/CaucasusRegionLID/ecapalang/ ${flags}; exec bash"
done

for lang in tt cv; do
    screen -dmS ${runname}_${lang} sh -c "CUDA_VISIBLE_DEVICES=0 python run.py --run_name ${runname}/${lang} --env_name test --lid_target ${lang} --feat_root /data1/CaucasusRegionLID/ecapalang/ ${flags}; exec bash"
done

for lang in ky hy-AM; do
    screen -dmS ${runname}_${lang} sh -c "CUDA_VISIBLE_DEVICES=1 python run.py --run_name ${runname}/${lang} --env_name test --lid_target ${lang} --feat_root /data1/CaucasusRegionLID/ecapalang/ ${flags}; exec bash"
done