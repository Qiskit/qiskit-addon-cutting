FROM quay.io/jupyter/minimal-notebook

LABEL maintainer="Jim Garrison <garrison@ibm.com>"

# The base notebook sets up a `work` directory "for backwards
# compatibility".  We don't need it, so let's just remove it.
RUN rm -rf work && \
    mkdir .src

COPY . .src/qiskit-addon-cutting

# Fix the permissions of ~/.src and ~/persistent-volume
USER root
RUN fix-permissions .src && \
    mkdir persistent-volume && fix-permissions persistent-volume
USER ${NB_UID}

# Consolidate the docs into the home directory
RUN mkdir docs && \
    cp -a .src/qiskit-addon-cutting/docs docs/qiskit-addon-cutting

# Pip install everything
RUN pip install -e '.src/qiskit-addon-cutting[notebook-dependencies]'
