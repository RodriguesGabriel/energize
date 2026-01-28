
FROM --platform=linux/amd64 ubuntu:24.04
RUN apt-get update
RUN apt-get install -y wget libjpeg8-dev zlib1g-dev locales jq && locale-gen en_US.UTF-8 
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV LC_ALL en_US.UTF-8
ENV APP_DIR /home/
WORKDIR ${APP_DIR}

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh 
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN echo $PATH
ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f environment.yml
RUN echo 'source activate /opt/conda/envs/energize/' >> ~/.bashrc

RUN export PATH=$CONDA_DIR/bin:$PATH
RUN conda init
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate energize
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> ~/.bashrc
COPY ./energize ./energize
COPY ./example ./example

# Activate conda environment and use the environment's python
ENV PATH="/opt/conda/envs/energize/bin:$PATH"

CMD ["/opt/conda/envs/energize/bin/python", "-u", "-m", "energize.main", "-d", "mnist", "-c", "example/example_config.json",  "-g", "example/energize.grammar", "--run", "0", "--gpu-enabled"]