#########################
Installation Instructions
#########################

There are two options: installing locally or using within a Docker
container.

- If you are using Linux or macOS with an Intel chip (i.e., not the
  new M1 or M2 chips), everything should work natively, so we
  recommend the first option.
- All users on ARM chips, as well as all Windows users, will have to
  use the toolbox within Docker (the second option) for everything to
  work as designed.

Option 1: Local installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **OPTIONAL** If a user wishes to use the circuit cutting tool to
   automatically find optimized wire cuts for a circuit too large for
   the free version of CPLEX, they should acquire a license and install
   the `full
   version <https://www.ibm.com/products/ilog-cplex-optimization-studio>`__.

-  Enter a Python environment and install the software

.. code:: sh

    git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
    cd circuit-knitting-toolbox
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install tox notebook -e '.[notebook-dependencies]'
    jupyter notebook

-  Navigate to the notebooks in the ``docs/tutorials/`` directory to run the
   tutorials.

Option 2: Use within Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have provided a `Dockerfile <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/blob/main/Dockerfile>`__, which can be used to
build a Docker image, as well as a
`docker-compose.yml <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/blob/main/docker-compose.yml>`__ file, which allows one
to use the Docker image with just a few simple commands. If you have
Docker installed but not `Docker
Compose <https://pypi.org/project/docker-compose/>`__, the latter can be
installed by first running ``pip install docker-compose``.

.. code:: sh

    git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
    cd circuit-knitting-toolbox
    docker-compose build
    docker-compose up

Depending on your system configuration, you may need to type ``sudo``
before each ``docker-compose`` command.

.. note::

   If you are instead using `podman <https://podman.io/>`_ and
   `podman-compose <https://github.com/containers/podman-compose>`_,
   the commands are:

   .. code:: sh

       podman machine start
       podman-compose --podman-pull-args short-name-mode="permissive" build
       podman-compose up

Once the container is running, you should see a message like this:

::

    notebook_1  |     To access the server, open this file in a browser:
    notebook_1  |         file:///home/jovyan/.local/share/jupyter/runtime/jpserver-7-open.html
    notebook_1  |     Or copy and paste one of these URLs:
    notebook_1  |         http://e4a04564eb39:8888/lab?token=00ed70b5342f79f0a970ee9821c271eeffaf760a7dcd36ec
    notebook_1  |      or http://127.0.0.1:8888/lab?token=00ed70b5342f79f0a970ee9821c271eeffaf760a7dcd36ec

Locate the *last* URL in your terminal (the one that includes
``127.0.0.1``), and navigate to that URL in a web browser to access the
Jupyter notebook interface.

The home directory includes a subdirectory named ``persistent-volume``.
All work you’d like to save should be placed in this directory, as it is
the only one that will be saved across different container runs.
