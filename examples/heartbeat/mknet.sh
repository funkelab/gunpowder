#!/usr/bin/env bash

NAME=$(basename "$PWD")

docker rm -f $NAME

USER_ID=${UID}
USER_HOME=${HOME}

echo "Starting as user ${USER_ID} with home ${HOME}"

NV_GPU=1 docker run --rm \
    -u ${USER_ID} \
    -e HOME=${USER_HOME} \
    -v ${PWD}:/run \
    -w /run \
    --name ${NAME} \
    funkey/gunpowder:v0.2 \
    python -u mknet.py
