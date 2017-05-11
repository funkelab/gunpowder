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
    git checkout 49c601e2d4f633ee510fc7b10e3d962bd9386363
ENV PYTHONPATH /src/augment:$PYTHONPATH

WORKDIR /src/dvision
RUN git clone -b v0.1.1 --depth 1 https://github.com/TuragaLab/dvision . && \
    pip install -r requirements.txt
ENV PYTHONPATH /src/dvision:$PYTHONPATH

# install gunpowder

WORKDIR /src/gunpowder
RUN git clone -b master https://github.com/TuragaLab/gunpowder .
ENV PYTHONPATH /src/gunpowder:$PYTHONPATH

# test the container

WORKDIR /run
ADD test_environment.py /run

# run a test
CMD ["python", "test_environment.py"]
