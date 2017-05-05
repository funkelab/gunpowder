#!/usr/bin/env bash

rm snapshots/*
rm net_iter*

NAME=$(basename "$PWD")

sudo nvidia-docker rm -f $NAME

USER_ID=${UID}
USER_HOME=${HOME}

echo "Starting as user ${USER_ID} with home ${HOME}"

NV_GPU=1 sudo nvidia-docker run --rm \
    -u ${USER_ID} \
    -e HOME=${USER_HOME} \
    -v ${PWD}:/run \
    -w /run \
    --name ${NAME} \
    funkey/gunpowder:latest \
    python -u train.py
