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
    screen -dmS ${runname}_${lang} 
    screen -S ${runname}_${lang} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
    screen -S ${runname}_${lang} -X stuff "CUDA_VISIBLE_DEVICES=0 python train_ood.py --ckpt_name ${runname}/${lang} --lid_target ${lang} --feat_root /data1/AfricanContinentLID/ecapalang/ ${flags}^M"
done

for lang in ckb kmr sr; do
    screen -dmS ${runname}_${lang} 
    screen -S ${runname}_${lang} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
    screen -S ${runname}_${lang} -X stuff "CUDA_VISIBLE_DEVICES=1 python train_ood.py --ckpt_name ${runname}/${lang} --lid_target ${lang} --feat_root /data1/AfricanContinentLID/ecapalang/ ${flags}^M"
done

for lang in tt cv; do
    screen -dmS ${runname}_${lang} 
    screen -S ${runname}_${lang} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
    screen -S ${runname}_${lang} -X stuff "CUDA_VISIBLE_DEVICES=0 python train_ood.py --ckpt_name ${runname}/${lang} --lid_target ${lang} --feat_root /data1/AfricanContinentLID/ecapalang/ ${flags}^M"
done

for lang in ky hy-AM; do
    screen -dmS ${runname}_${lang} 
    screen -S ${runname}_${lang} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
    screen -S ${runname}_${lang} -X stuff "CUDA_VISIBLE_DEVICES=1 python train_ood.py --ckpt_name ${runname}/${lang} --lid_target ${lang} --feat_root /data1/AfricanContinentLID/ecapalang/ ${flags}^M"
done