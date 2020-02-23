Installation
============

We recommend using conda_.

First, clone this repo

.. code-block:: shell

    git clone git@github.com:shane-d-zhang/PyModule.git

Then create an environment:

.. code-block:: shell

    cd PyModule
    conda env create -f environment.yml
    conda activate pymodule

After installing dependencies, or if you decide to use pip_
to install dependencies, run the follows to install `pymodule`
and missing dependencies:

.. code-block:: shell

    pip install -e .


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda
.. _pip: https://pip.pypa.io/en/stable/
