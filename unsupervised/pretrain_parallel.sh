#!/bin/sh
runname=$1
flags=$2
echo $runname
echo $flags

rm -rf unsupervised/output/${runname}
mkdir unsupervised/output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

for env in rm1_mc20 rm2_mc16 rm3_mc16 rm4_mc20; do
    screen -dmS _${runname}_${env}_ sh -c "CUDA_VISIBLE_DEVICES=0 python unsupervised/train_unsup.py --ckpt_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
done

for env in apartment_mc19 hotel_mc19 office_mc13; do
    screen -dmS _${runname}_${env}_ sh -c "CUDA_VISIBLE_DEVICES=1 python unsupervised/train_unsup.py --ckpt_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
done