#############################################
CutQC (legacy circuit cutting implementation)
#############################################

The ``cutqc`` module implements the wire cutting technique and
automatic cut finding method described in the research paper
`arXiv:2012.02333 <https://arxiv.org/abs/2012.02333>`_.  Historically,
this was the original circuit cutting implementation in the Circuit
Knitting Toolbox.  Going forward, this module is deprecated.  Users
of the toolbox should use the new circuit cutting interface, instead.

.. _cutqc tutorials:

.. include:: README.rst

.. nbgallery::
    :glob:

    tutorials/*


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
