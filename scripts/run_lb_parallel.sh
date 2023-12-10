#!/bin/sh
runname=$1
flags=$2
device=$3
echo $runname
echo $flags
echo $device

rm -rf output/${runname}
mkdir output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

for env in apartment_mc19 hotel_mc19 office_mc13; do
    if [ -z $device ]
    then
        screen -dmS ${runname}_${env} sh -c "python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
    else
        screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=${device} python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
    fi
done
