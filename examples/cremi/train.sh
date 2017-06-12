rm snapshots/*

export NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

NV_GPU=0 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.2-prerelease \
    python -u train.py
