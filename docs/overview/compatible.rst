Compatible Libraries
====================

A list of Python libraries, beyond just `numpy <http://numpy.org/>`_ and `pandas <http://pandas.pydata.org/>`_, providing
mathematical algorithms that are compatible with grapp's :py:class:`LinearOperator` interface.

scipy
-----

Many of the functions in the `scipy.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_ module
are compatible with ``LinearOperator``. Of particular note are:

 * The conjugate gradient method `scipy.sparse.linalg.cg <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.cg>`_
 * Least squares solver `scipy.sparse.linalg.lsqr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.lsqr>`_
 * Eigenvalue decomposition via `scipy.sparse.linalg.eigs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.eigs>`_
 * The preconditioned eigensolver `scipy.sparse.linalg.lobpcg <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.lobpcg>`_


PyLops
------

`PyLops <http://pylops.readthedocs.io/>`_ is a library focused on matrix-free methods, primarily focused on signal processing, but with some general purpose methods as well.
The grapp ``LinearOperator`` classes are compatible with PyLops. Of particular note are:

 * The HutchPP estimator for the trace of a matrix, `pylops.utils.estimators.trace_hutchpp <https://pylops.readthedocs.io/en/stable/api/generated/pylops.utils.estimators.trace_hutchpp.html#pylops.utils.estimators.trace_hutchpp>`_
 * Combining multiple ``LinearOperator`` via `pylops.HStack <https://pylops.readthedocs.io/en/stable/api/generated/pylops.HStack.html>`_, `pylops.VStack <https://pylops.readthedocs.io/en/stable/api/generated/pylops.VStack.html>`_, and `pylops.BlockDiag <https://pylops.readthedocs.io/en/stable/api/generated/pylops.BlockDiag.html>`_
 * Kronecker product of ``LinearOperator`` via `pylops.Kronecker <https://pylops.readthedocs.io/en/stable/api/generated/pylops.Kronecker.html>`_
 * Various optimization solvers with forced sparsity, such as `pylops.optimization.cls_sparsity.FISTA <https://pylops.readthedocs.io/en/stable/api/generated/pylops.optimization.cls_sparsity.FISTA.html>`_

spgl1
-----

`spgl1 <https://spgl1.readthedocs.io/en/latest/>`_ is a regularized least-squares solver (e.g., LASSO) that is compatible with ``LinearOperator``.
