FROM jupyter/minimal-notebook:python-3.11

LABEL maintainer="Jim Garrison <garrison@ibm.com>"

# The base notebook sets up a `work` directory "for backwards
# compatibility".  We don't need it, so let's just remove it.
RUN rm -rf work && \
    mkdir .src

COPY . .src/circuit-knitting-toolbox

# Fix the permissions of ~/.src and ~/persistent-volume
USER root
RUN fix-permissions .src && \
    mkdir persistent-volume && fix-permissions persistent-volume
USER ${NB_UID}

# Consolidate the docs into the home directory
RUN mkdir docs && \
    cp -a .src/circuit-knitting-toolbox/docs docs/circuit-knitting-toolbox

# Pip install everything
RUN pip install -e '.src/circuit-knitting-toolbox[notebook-dependencies]'
