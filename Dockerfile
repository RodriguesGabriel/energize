FROM ubuntu:22.04
RUN apt-get update \
    && apt-get install -y python3.10 python3-pip wget libjpeg8-dev zlib1g-dev locales && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL en_US.UTF-8
ENV APP_DIR /home/
WORKDIR ${APP_DIR}
RUN pip install --upgrade pip

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN echo 'source activate /opt/conda/envs/energize/' >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

RUN export PATH=$CONDA_DIR/bin:$PATH
RUN source activate /opt/conda/envs/energize/
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> ~/.bashrc