#!/bin/sh
runname=$1
flags=$2
echo $runname
echo $flags

rm -rf output/${runname}
mkdir output/${runname}
rm -rf checkpoints/${runname}
mkdir checkpoints/${runname}

# for env in rm1_mc20 rm2_mc16 rm3_mc16 rm4_mc20; do
#     screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=0 python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
# done

# for env in apartment_mc19 hotel_mc19 office_mc13; do
#     screen -dmS ${runname}_${env} sh -c "CUDA_VISIBLE_DEVICES=1 python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}; exec bash"
# done

for env in rm1_mc20 rm2_mc16 rm3_mc16 rm4_mc20; do
    screen -dmS ${runname}_${env} 
    screen -S ${runname}_${env} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
    screen -S ${runname}_${env} -X stuff "CUDA_VISIBLE_DEVICES=0 python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}^M"
done

for env in apartment_mc19 hotel_mc19 office_mc13; do
    screen -dmS ${runname}_${env} 
    screen -S ${runname}_${env} -X stuff "source ~/oalcf3.11.7/bin/activate^M"
    screen -S ${runname}_${env} -X stuff "CUDA_VISIBLE_DEVICES=1 python run.py --run_name ${runname}/${env} --env_name ${env} ${flags}^M"
done