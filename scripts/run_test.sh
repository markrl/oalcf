#!/bin/sh
runname=$1
flags=$2
echo $runname
echo $flags

rm -rf output/${runname}
mkdir output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

for env in rm1_mc14 rm2_mc14 rm3_mc14 rm4_mc14; do
    screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=0 python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
done

for env in apartment_mc04 hotel_mc04 office_mc04; do
    screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=1 python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
done