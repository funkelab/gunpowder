FROM funkey/caffe
LABEL maintainer jfunke@iri.upc.edu

# install dependencies for gunpowder
ENV MALIS_ROOT=/src/malis
ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
ENV MALIS_REVISION=a1e084b0e0fec266c454431d786ac36b8ab6fe96

ENV AUGMENT_ROOT=/src/augment
ENV AUGMENT_REPOSITORY=https://github.com/funkey/augment.git
ENV AUGMENT_REVISION=4a42b01ccad7607b47a1096e904220729dbcb80a 

ENV DVISION_ROOT=/src/dvision
ENV DVISION_REPOSITORY=https://github.com/TuragaLab/dvision.git
ENV DVISION_REVISION=v0.1.1

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
