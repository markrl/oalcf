#!/bin/sh
runname=$1

screen -ls | grep "${runname}" | awk '{print $1}' | while read -r env; do
    screen -S $env -X quit
done