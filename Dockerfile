FROM continuumio/miniconda3

WORKDIR /src/DeepR

COPY environment.yml /src/DeepR/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/DeepR

RUN pip install --no-deps -e .
