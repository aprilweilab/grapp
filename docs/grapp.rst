grapp API
=========

.. automodule:: grapp
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

Linear Algebra
--------------

Non-standardized Linear Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These operators work with `scipy.sparse.linalg`, and provide the ability to do matrix products
against the unmodified genotype matrix that is represented by a GRG.

.. autoclass:: grapp.linalg.ops_scipy.SciPyXOperator
.. autoclass:: grapp.linalg.ops_scipy.SciPyXTXOperator
.. autoclass:: grapp.linalg.ops_scipy.SciPyXXTOperator

Standardized Linear Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These operators work with `scipy.sparse.linalg`, and provide the ability to do matrix products
against the genotype matrix that is represented by a GRG, except that genotype matrix is
implicitly standardized by subtracting the mean and dividing by the standard deviation.

.. autoclass:: grapp.linalg.ops_scipy.SciPyStdXOperator
.. autoclass:: grapp.linalg.ops_scipy.SciPyStdXTXOperator
.. autoclass:: grapp.linalg.ops_scipy.SciPyStdXXTOperator

Linear Operators for Multiple GRGs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These operators are the same as the ones above, except they allow for multiple GRGs to be used
for a single product. For example, if you have GRGs for each of the 22 autosomes, you can construct
these operators and pass in all 22 GRGs. The resulting operator will perform matrix multiplications
against the entire autosome. You can use multiple threads to parallelize by GRG.

.. autoclass:: grapp.linalg.ops_scipy.MultiSciPyXOperator
.. autoclass:: grapp.linalg.ops_scipy.MultiSciPyXTXOperator
.. autoclass:: grapp.linalg.ops_scipy.MultiSciPyStdXOperator
.. autoclass:: grapp.linalg.ops_scipy.MultiSciPyStdXTXOperator

PCA and other helper methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grapp.linalg
   :members:
   :undoc-members:
   :show-inheritance:

Association Studies (GWAS)
--------------------------

.. automodule:: grapp.assoc
   :members:
   :undoc-members:
   :show-inheritance:

Nearest Neighbor Comparisons
-----------------------------

.. automodule:: grapp.nn
   :members:
   :undoc-members:
   :show-inheritance:

Filtering, Export, etc.
-----------------------------

Filtering GRGs
~~~~~~~~~~~~~~

.. automodule:: grapp.util.filter
   :members:
   :undoc-members:
   :show-inheritance:

Exporting to IGD
~~~~~~~~~~~~~~~~

.. automodule:: grapp.util.igd
   :members:
   :undoc-members:
   :show-inheritance:

Simple Calculations
~~~~~~~~~~~~~~~~~~~

.. automodule:: grapp.util.simple
   :members:
   :undoc-members:
   :show-inheritance: