FROM nvidia/cuda:11.2.2-devel-ubuntu18.04

MAINTAINER zdchen <chenzd15thu@ucsb.edu>

RUN mkdir projects
WORKDIR projects

# Install Sputnik Dependencies
RUN apt-get -y update --fix-missing
RUN apt-get install -y git emacs wget libgoogle-glog-dev
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y cmake

# Build sputnik
RUN git clone https://github.com/google-research/sputnik.git
WORKDIR sputnik
RUN mkdir build
WORKDIR build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="70"
RUN make -j12

ENV CUDA_INSTALL_PATH=/usr/local/cuda-11.2
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=/projects/sputnik/build/sputnik/

# install Python3.8
RUN apt-get install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.8 python3-pip

# install python libraries
RUN pip3 install numpy
RUN pip3 install matplotlib

# install nsight compute
RUN apt-get update -y
RUN apt-get -y install nsight-compute-2020.3.1

WORKDIR /

ENV SPUTNIK_PATH=/projects/sputnik
ENV NCU_PATH=/opt/nvidia/nsight-compute/2020.3.1/ncu

WORKDIR /projects