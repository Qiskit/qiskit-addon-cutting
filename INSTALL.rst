Installation Instructions
=========================

Let's get started with the Circuit Knitting Toolbox (CKT)! The first
thing to do is choose how you're going to run and install the
packages. There are three primary ways to do this: :ref:`Option 1`,
:ref:`Option 2`, or :ref:`Option 3`.


Pre-Installation
^^^^^^^^^^^^^^^^
Users with ARM chips and Windows users should consult the
:ref:`Platform Support` section to determine which installation option
is appropriate for them. Users who wish to run within a
containerized environment may skip the pre-installation and move straight
to :ref:`Option 3`.

Users who wish to install locally or via PyPI may follow a few set of
common instructions to prepare for installation:

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh
    
    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh
    
    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, use the following commands in PowerShell:

.. code:: sh
    
    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1

.. note::

    **OPTIONAL** If a user wishes to use the circuit cutting tool to
    automatically find optimized wire cuts for a circuit too large for
    the free version of CPLEX, they should acquire a license and install
    the `full
    version <https://www.ibm.com/products/ilog-cplex-optimization-studio>`__.


.. _Option 1:

Option 1: pip installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Upgrade pip.

.. code:: sh
    
    pip install --upgrade pip

Install the CKT package.

.. code:: sh

    pip install circuit-knitting-toolbox

Users intending to use the entanglement forging tool should install the pyscf option.

.. code:: sh
    
    pip install 'circuit-knitting-toolbox[pyscf]'

Users intending to use the automatic cut finding functionality in the CutQC package should install the cplex option.

.. code:: sh
    
    pip install 'circuit-knitting-toolbox[cplex]'
    

.. _Option 2:

Option 2: Local Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the CKT repository.

.. code:: sh

    git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
    
Upgrade pip and enter the repository. 

.. code:: sh
    
    pip install --upgrade pip
    cd circuit-knitting-toolbox

Install CKT from source. Install the notebook dependencies in order to run
all the visualizations in the tutorial notebooks.

.. code:: sh
    
    pip install tox notebook -e '.[notebook-dependencies]'

Users intending to use the entanglement forging tool should install the pyscf option.

.. code:: sh
    
    pip install '.[pyscf]'

Users intending to use the automatic cut finding functionality in the CutQC package should install the cplex option.

.. code:: sh
    
    pip install -e '.[cplex]'


.. _Option 3:

Option 3: Use within Docker
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
Jupyter Notebook interface.

The home directory includes a subdirectory named ``persistent-volume``.
All work you’d like to save should be placed in this directory, as it is
the only one that will be saved across different container runs.


Running some Examples
^^^^^^^^^^^^^^^^^^^^^
From inside the ``circuit_knitting_toolbox`` repository, open a `Jupyter Notebook <https://jupyter.org/install>`__, navigate
to the tutorials, and open a Jupyter Notebook instance.

.. code::
    
    cd docs/<circuit_cutting | entanglement_forging>/tutorials
    jupyter notebook


.. _Platform Support:

Platform Support
^^^^^^^^^^^^^^^^

Users of Mac M1 or M2 chips and Windows users may have issues running certain components of CKT.

- If you are using Linux or macOS with an Intel chip (i.e., not the
  new M1 or M2 chips), everything should work natively, so we
  recommend either :ref:`Option 1` or :ref:`Option 2`.
- All users on ARM chips, as well as all Windows users, may have to
  use the toolbox within Docker (:ref:`Option 3`), depending on what tools they intend to use.
    - The automatic wire cut search in the ``circuit_cutting`` module depends
      on cplex, which is only available on Intel chips and is not yet available
      for Python 3.11.
    - The entanglement forging tool requires pyscf, which does not support Windows.
