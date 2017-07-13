#!/usr/bin/env bash

rm snapshots/*
rm net_iter*

NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

USER_ID=${UID}
echo "Starting as user ${USER_ID}"

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -w /workspace \
    --name ${NAME} \
    funkey/gunpowder:v0.2 \
    python -u train_synapses.py"





