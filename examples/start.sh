rm snapshots/*

sudo mount --make-shared /nrs/turaga

export NAME=$(basename "$PWD")

nvidia-docker rm $NAME

NV_GPU=0 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nrs/turaga:/nrs/turaga:shared \
    -v $HOME/src/gunpowder:/opt/gunpowder \
    -v $HOME/src/augment:/opt/augment \
    --name $NAME \
    turagalab/greentea:cuda8.0-cudnn6-caffe_gt-2017.04.17-pygt-0.9.4b \
    bash -c 'PYTHONPATH=$PYTHONPATH/opt/gunpowder:/opt/augment python -u cremi.py'
