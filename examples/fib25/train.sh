rm snapshots/*
rm net_iter*

NAME=$(basename "$PWD")

# docker build \
#     /groups/turaga/home/grisaitisw/src/gunpowder/docker/grisaitis_mod \
#     -t grisaitisw/gunpowder:latest

nvidia-docker rm -f $NAME

WILLIAM_USER_ID=53850
WILLIAM_HOME=/groups/turaga/home/grisaitisw

NV_GPU=1 nvidia-docker run --rm \
    -u $WILLIAM_USER_ID \
    -e "HOME=$WILLIAM_HOME" \
    -v /groups/turaga/home:/groups/turaga/home \
    -v $WILLIAM_HOME/src/gunpowder:/src/gunpowder \
    -w $(pwd) \
    --name $NAME \
    grisaitisw/gunpowder:latest \
    /bin/bash -c "PYTHONPATH=$WILLIAM_HOME/src/dvision:\$PYTHONPATH && python -u train.py"

#    /bin/bash -c "PYTHONPATH=~/src/dvision:\$PYTHONPATH && python -c 'import sys; from pprint import pprint; pprint(sys.path)'"
#    /bin/bash -c "PYTHONPATH=~/src/gunpowder:~/src/dvision:\$PYTHONPATH && python -c 'import gunpowder; print gunpowder.__file__'"
#    /bin/bash -c "PYTHONPATH=~/src/gunpowder:~/src/dvision:\$PYTHONPATH && python -u train.py"
#    /bin/bash -c "PYTHONPATH=~/src/gunpowder:\$PYTHONPATH && env && ls -l ~ && pwd && ls -l"

#todo: fix docker image
#error:
#ImportError: /opt/conda/lib/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /opt/caffe/python/caffe/_caffe.so)
