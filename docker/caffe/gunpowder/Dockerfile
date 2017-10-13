FROM funkey/caffe
LABEL maintainer jfunke@iri.upc.edu

# install dependencies for gunpowder
ENV MALIS_ROOT=/src/malis
ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
ENV MALIS_REVISION=e92f758725e0a2067cacd39827fad87f1848a2cb

ENV AUGMENT_ROOT=/src/augment
ENV AUGMENT_REPOSITORY=https://github.com/funkey/augment.git
ENV AUGMENT_REVISION=4a42b01ccad7607b47a1096e904220729dbcb80a 

ENV DVISION_ROOT=/src/dvision
ENV DVISION_REPOSITORY=https://github.com/TuragaLab/dvision.git
ENV DVISION_REVISION=v0.1.1

ENV WATERZ_ROOT=/src/waterz
ENV WATERZ_REPOSITORY=https://github.com/funkey/waterz
ENV WATERZ_REVISION=d2bede846391c56a54365c13d5b2f2f4e6db4ecd

WORKDIR ${MALIS_ROOT}
RUN git clone ${MALIS_REPOSITORY} . && \
    git checkout ${MALIS_REVISION}
RUN python setup.py build_ext --inplace
ENV PYTHONPATH ${MALIS_ROOT}:$PYTHONPATH

WORKDIR ${AUGMENT_ROOT} 
RUN git clone ${AUGMENT_REPOSITORY} . && \
    git checkout ${AUGMENT_REVISION}
RUN pip install -r requirements.txt
ENV PYTHONPATH ${AUGMENT_ROOT}:$PYTHONPATH

WORKDIR ${DVISION_ROOT}
RUN git clone -b ${DVISION_REVISION} --depth 1 ${DVISION_REPOSITORY} .
RUN pip install -r requirements.txt
ENV PYTHONPATH ${DVISION_ROOT}:$PYTHONPATH

WORKDIR ${WATERZ_ROOT}
RUN git clone ${WATERZ_REPOSITORY} . && \
    git checkout ${WATERZ_REVISION}
RUN pip install cython
RUN mkdir -p /.cython/inline
ENV PYTHONPATH ${WATERZ_ROOT}:$PYTHONPATH

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
