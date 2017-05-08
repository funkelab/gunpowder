FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer jfunke@iri.upc.edu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/src/caffe
ENV CAFFE_REPOSITORY=https://github.com/naibaf7/caffe.git
ENV CAFFE_REVISION=6c559baca26c6436d277839292894c1e3f2549a0

WORKDIR $CAFFE_ROOT
RUN git clone ${CAFFE_REPOSITORY} . && \
    git checkout ${CAFFE_REVISION}
RUN pip install --upgrade pip && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done

WORKDIR $CAFFE_ROOT/build
RUN cmake -DUSE_INDEX_64=1 -DUSE_CUDA=1 -DUSE_LIBDNN=1 -DUSE_CUDNN=1 -DUSE_OPENMP=0 -DUSE_GREENTEA=0 .. && \
    make --jobs $(nproc)

# setup env to find pycaffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    ldconfig

# install dependencies for gunpowder

WORKDIR /src/malis
RUN git clone https://github.com/TuragaLab/malis . && \
    git checkout a1e084b0e0fec266c454431d786ac36b8ab6fe96 && \
    python setup.py build_ext --inplace
ENV PYTHONPATH /src/malis:$PYTHONPATH

WORKDIR /src/augment
RUN git clone https://github.com/funkey/augment . && \
    git checkout 49c601e2d4f633ee510fc7b10e3d962bd9386363
ENV PYTHONPATH /src/augment:$PYTHONPATH

WORKDIR /src/dvision
RUN git clone -b v0.1.1 --depth 1 https://github.com/TuragaLab/dvision . && \
    pip install -r requirements.txt
ENV PYTHONPATH /src/dvision:$PYTHONPATH

# install gunpowder

WORKDIR /src/gunpowder
ENV GUNPOWDER_REVISION=09fb08e3d0c024b794f477343009cb3a4b7ffcc4
RUN git clone https://github.com/TuragaLab/gunpowder . && \
    git checkout ${GUNPOWDER_REVISION}
ENV PYTHONPATH /src/gunpowder:$PYTHONPATH

# test the container

WORKDIR /run
ADD test_environment.py /run

# run a test
CMD ["python", "test_environment.py"]
