rm snapshots/*

sudo mount --make-shared /nrs/turaga

export NAME=$(basename "$PWD")

nvidia-docker rm $NAME

NV_GPU=0 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nrs/turaga:/nrs/turaga:shared \
    --name $NAME \
    funkey/gunpowder:latest \
    bash -c 'PYTHONPATH=$PYTHONPATH:/opt/gunpowder:/opt/augment python -u cremi.py'
