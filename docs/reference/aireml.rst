aireml API
==========

``aireml`` is a standalone package (it does not import ``grapp`` or ``pygrgl``)
implementing average-information REML for variance components whose covariance
structure is supplied as a ``scipy.sparse.linalg.LinearOperator``. It follows
`Lee et al. (2026) <https://doi.org/10.1093/genetics/iyag074>`_, "Genetic
prediction with ARG-powered linear algebra".

Because the relatedness matrix is only ever multiplied by vectors, any of
``grapp``'s GRG-backed linear operators can be used as the genetic covariance,
without the ``N x N`` matrix ever being formed::

   from aireml import fit_reml, grm_from_genotypes
   from grapp.linalg.ops_scipy import SciPyStdXOperator

   genotypes = SciPyStdXOperator(grg)          # N x M, backed by the GRG
   result = fit_reml(phenotypes, grm_from_genotypes(genotypes))
   print(result.summary())

Fitting
-------

.. autofunction:: aireml.fit_reml
.. autofunction:: aireml.haseman_elston
.. autoclass:: aireml.REMLResult
   :members:

Operators
---------

.. autofunction:: aireml.grm_from_genotypes
.. autofunction:: aireml.centered_operator
.. autofunction:: aireml.as_symmetric_operator
.. autofunction:: aireml.identity_operator
.. autofunction:: aireml.materialize
.. autoclass:: aireml.LinearCombinationOperator
   :members:

Linear algebra
--------------

.. autofunction:: aireml.estimate_trace
.. autoclass:: aireml.ConjugateGradientSolver
   :members:
.. autoclass:: aireml.DenseSolver
   :members:
.. autoclass:: aireml.NystromSketch
   :members:
.. autoclass:: aireml.RemlState
   :members:
