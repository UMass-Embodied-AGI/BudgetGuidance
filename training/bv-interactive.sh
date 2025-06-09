#!/bin/bash
for arg in "$@"; do
    eval "$arg"
done

echo "g:   ${g:=8}"

export job_name=granite-ctrl-llm

bsub \
    -J $job_name \
    -q interactive \
    -gpu \"num=$g/task:mode=exclusive_process\" \
    -n 1 \
    -Is \
    -M 512G \
    -W 60:00 \
        bash
    #-q standard \
