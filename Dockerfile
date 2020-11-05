FROM nvidia/cuda
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    cmake \
    tmux \
    htop \
    nano \
    python3.7 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    tar \
    git \
    gcc

ENV PYTHONPATH /locality_experiments
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . /locality_experiments
