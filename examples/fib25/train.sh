#!/usr/bin/env bash

CONTAINER=funkey/gunpowder:v0.3-prerelease

NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

USER_ID=${UID}
USER_HOME=${HOME}

echo "Starting as user ${USER_ID} with home ${HOME}"

nvidia-docker pull ${CONTAINER}

NV_GPU=1 nvidia-docker run --rm \
    -u ${USER_ID} \
    -e HOME=${USER_HOME} \
    -v ${PWD}:/run \
    -w /run \
    --name ${NAME} \
    ${CONTAINER} \
    python -u train.py 400000 0
