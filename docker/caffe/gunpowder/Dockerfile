FROM caffe
LABEL maintainer jfunke@iri.upc.edu

# install dependencies for gunpowder

WORKDIR /src/malis
RUN git clone https://github.com/TuragaLab/malis . && \
    git checkout a1e084b0e0fec266c454431d786ac36b8ab6fe96 && \
    python setup.py build_ext --inplace
ENV PYTHONPATH /src/malis:$PYTHONPATH

WORKDIR /src/augment
RUN git clone https://github.com/funkey/augment . && \
    git checkout 4a42b01ccad7607b47a1096e904220729dbcb80a && \
    pip install -r requirements.txt
ENV PYTHONPATH /src/augment:$PYTHONPATH

WORKDIR /src/dvision
RUN git clone -b v0.1.1 --depth 1 https://github.com/TuragaLab/dvision . && \
    pip install -r requirements.txt
ENV PYTHONPATH /src/dvision:$PYTHONPATH

# install gunpowder

WORKDIR /src/gunpowder
RUN git clone https://github.com/TuragaLab/gunpowder . && \
    git checkout d2dec21fded1a444a1b6e04a67d3e5060fd62b13 && \
    pip install -r requirements.txt
ENV PYTHONPATH /src/gunpowder:$PYTHONPATH

# test the container

WORKDIR /run
ADD test_environment.py /run

# run a test
CMD ["python", "test_environment.py"]
