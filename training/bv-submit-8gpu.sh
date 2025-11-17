#!/bin/bash

export job_name=granite-ctrl-llm-8g

bsub \
        -J $job_name \
        -gpu \"num=8/task:mode=exclusive_process\" \
        -n 1 \
        -M 512G \
        -G grp_inference_scaling \
        -W 120:00 \
        -o ./bv_output/${job_name}-%J-$1.stdout \
        -e ./bv_output/${job_name}-%J-$1.stderr \
        < $1

exit 0
        #-q standard \
