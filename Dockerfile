FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && \ 
        apt-get -y install git
RUN apt-get -y install git-lfs
RUN apt-get -y install openssh-client

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
        micromamba clean --all --yes

# Directory for sources
RUN mkdir $HOME/source/ && \
        cd $HOME/source

# make sure to use https instead of ssh for cloning
RUN git config --global url.https://github.com/.insteadOf git@github.com: && \
        git config --global --add safe.directory '/tmp/*'
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://github.com/alabamagan/mri_radiomics_toolkit.git
RUN chown $MAMBA_USER:$MAMBA_USER -R $HOME/source/ 
RUN cd mri_radiomics_toolkit/ && git checkout v0.1

# Install requirements
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN cd mri_radiomics_toolkit/ThirdParty/mnts && pip install ./
RUN cd mri_radiomics_toolkit/ThirdParty/RENT && pip install . 
RUN cd mri_radiomics_toolkit/ && pip install .
RUN pip install pytest

# Install current sourcecode
COPY --chown=$MAMBA_USER:$MAMBA_USER src /tmp/src
COPY --chown=$MAMBA_USER:$MAMBA_USER setup* /tmp
RUN cd /tmp/ && echo `ls` && pip install .

ARG MAMBA_DOCKERFILE_ACTIVATE=0