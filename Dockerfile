FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

WORKDIR /home/deepr

COPY . .

RUN conda install -c conda-forge gcc python=${PYTHON_VERSION} && \
    conda env update -n base -f environment_CUDA.yml

RUN pip install --no-deps -e .
