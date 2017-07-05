FROM funkey/caffe
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

# assumes that gunpowder package directory and requirements.txt are in build
# context (the complementary Makefile ensures that)
ADD gunpowder /src/gunpowder/gunpowder
ADD requirements.txt /src/gunpowder/requirements.txt
WORKDIR /src/gunpowder
RUN pip install -r requirements.txt
ENV PYTHONPATH /src/gunpowder:$PYTHONPATH

# test the container

WORKDIR /run
ADD test_environment.py /run

# run a test
CMD ["python", "test_environment.py"]
