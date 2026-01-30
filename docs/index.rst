grapp Documentation
===================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   API Reference <grapp>

grapp is a Python library for performing highly scalable, highly efficient calculations on
(GRG) files. `GRG is a graph-based data structure and file format format <https://grgl.readthedocs.io/en/stable/concepts.html>`_.
GRG can perform complex data calculations orders of magnitude faster than other methods.

grapp is both a tool set and a framework:

* The *tool set* includes features like filtering (samples and mutations), PCA, GWAS with covariates, phenotype simulation (via `grg_pheno_sim <https://github.com/aprilweilab/grg_pheno_sim>`_), and data export.
* The *framework* can be used to build tools/methods for statistical and population genetics. The framework contains Python functionality for integrating GRGs with `scipy <https://scipy.org/>`_ and other numerical libraries. In particular, the ``LinearOperator`` functionality can interoperate with many functions in `scipy.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#module-scipy.sparse.linalg>`_

Both integrate nicely with the Python data analysis ecosystem of `numpy <https://numpy.org>`_, `pandas <https://pandas.pydata.org>`_, and `scipy <https://scipy.org>`_.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
