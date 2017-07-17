#!/usr/bin/env bash

NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

USER_ID=${UID}
USER_HOME=${HOME}

echo "Starting as user ${USER_ID} with home ${HOME}"

NV_GPU=1 nvidia-docker run --rm \
    -u ${USER_ID} \
    -e HOME=${USER_HOME} \
    -v ${PWD}:/run \
    -w /run \
    --name ${NAME} \
    funkey/gunpowder:v0.2 \
    python -u process.py 0
