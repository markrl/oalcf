#!/bin/sh
runname=$1

for env in rm1_mc20 rm2_mc16 rm3_mc16 rm4_mc20; do
    screen -S ${runname}_${env} -X quit
done

for env in apartment_mc19 hotel_mc19 office_mc13; do
    screen -S ${runname}_${env} -X quit
done