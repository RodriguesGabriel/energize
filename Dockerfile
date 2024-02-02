FROM ubuntu:23.10
RUN apt-get update
RUN apt-get install -y wget libjpeg8-dev zlib1g-dev locales && locale-gen en_US.UTF-8 
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV LC_ALL en_US.UTF-8
ENV APP_DIR /home/
WORKDIR ${APP_DIR}

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh 
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN echo 'source activate /opt/conda/envs/energize/' >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

RUN export PATH=$CONDA_DIR/bin:$PATH
RUN source activate /opt/conda/envs/energize/
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> ~/.bashrc