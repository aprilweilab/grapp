import pygrgl
import numpy
from numpy.typing import NDArray
from typing import List


class NearestNeighborContext:
    """
    The main class for performing neighbor queries against the GRG format. Holds cached
    information related to nearest-neighbor queries on a specific GRG.
    """

    def __init__(self, grg: pygrgl.GRG):
        self._grg = grg
        self._muts_above = None
        self._samps_below = None

    @property
    def grg(self):
        return self._grg

    @property
    def muts_above(self) -> NDArray:
        """
        Vector of length grg.num_nodes, where each node's value is the number of Mutations above that node
        in the graph.
        """
        if self._muts_above is None:
            # One-time calculation of how many mutations above each node.
            self._muts_above = pygrgl.matmul(
                self.grg,
                numpy.ones((1, self.grg.num_mutations), dtype=numpy.int32),
                pygrgl.TraversalDirection.DOWN,
                emit_all_nodes=True,
            )[0]
        assert self._muts_above is not None
        return self._muts_above

    @property
    def samps_below(self) -> NDArray:
        """
        Vector of length grg.num_nodes, where each node's value is the number of sample nodes below that
        node in the graph.
        """
        if self._samps_below is None:
            # One-time calculation of how many samples beneath each node.
            self._samps_below = pygrgl.matmul(
                self.grg,
                numpy.ones((1, self.grg.num_samples), dtype=numpy.int32),
                pygrgl.TraversalDirection.UP,
                emit_all_nodes=True,
            )[0]
        assert self._samps_below is not None
        return self._samps_below

    def exact_hamming_dists(
        self,
        seeds: numpy.typing.NDArray,
        direction: pygrgl.TraversalDirection,
        emit_all_nodes: bool = False,
    ) -> numpy.typing.NDArray:
        """
        Using exact computations, get the Hamming distances from the matrix of input seeds
        to every other sample in the GRG.

        :param seeds: A two-dimensional numpy array. Each row corresponds to a single "query" for distances,
            and contains a '1' for every mutation (downward direction) or sample (upward direction) that is
            used by the query item.
        :type seeds: numpy.ndarray
        :param direction: Whether to find the distances to Samples (pygrgl.TraversalDirection.DOWN) or the distances
            to Mutations (pygrgl.TraversalDirection.UP). The number of columns in the seeds input matrix must match
            the direction, so columns(seeds) == grg.num_mutations if direction is down, and columns(seeds) == grg.num_samples
            if direction is up.
        :type direction: pygrgl.TraversalDirection
        :return: A two-dimensional numpy array where the number of rows matches the input matrix; i.e. each row is a result
            from each query. The number of columns is the opposite of the input (similar to pygrgl.matmul), so if the seeds
            have grg.num_mutations columns then the result will have grg.num_samples columns.
        :rtype: numpy.ndarray
        """
        assert seeds.ndim == 2, "Expected two-dimensional array as input"
        # This computes the (|y| - 2*|x ^ y|) part of the Hamming distance, where "x" is the query sample
        # and "y" is every other sample.
        rows = seeds.shape[0]
        if direction == pygrgl.TraversalDirection.DOWN:
            incols = self.grg.num_mutations
        else:
            assert direction == pygrgl.TraversalDirection.UP
            incols = self.grg.num_samples
        assert seeds.shape[1] == incols, f"Unexpected number of input columns: {incols}"

        directional_input = numpy.ones((rows, incols), dtype=numpy.int32) - (2 * seeds)
        hamming_result = pygrgl.matmul(
            self.grg, directional_input, direction, emit_all_nodes
        )
        # Finally, we need to add the |x| part of the distance. Since this is a constant, we could leave it off
        # when we only need nearest neighbors and not the actual distance value.
        for i in range(hamming_result.shape[0]):
            hamming_result[i, :] += sum(seeds[i] > 0)
        return hamming_result

    def exact_hamming_dists_by_sample(
        self,
        sample_ids: List[int],
        emit_all_nodes: bool = False,
    ) -> numpy.typing.NDArray:
        """
        Using exact computations, get the Hamming distances from the list of input sample IDs (for samples in
        the GRG) to every other sample in the GRG.

        :param sample_ids: List of GRG Node IDs for sample nodes, each of which will be queried for distance
            to all other samples.
        :type sample_ids: List[int]
        :param emit_all_nodes: Set to True to compute distances to every _node_ in the graph, not just every
            other sample. The output Matrix will have num_nodes columns when True.
        :type emit_all_nodes: bool
        :return: Matrix of distances, where each row corresponds to input sample IDs, and each column is the
            distance from the "other" sample ID. For example, if the 0th input is sample ID "n0", then the
            0th row of output is [D(n0, 0), D(n0, 1), ..., D(n0, N-1)] where N is the number of haploid samples.
        :rtype: numpy.ndarray
        """
        rows = len(sample_ids)
        assert rows > 0
        # This computes a vector of 1's for every mutation that is contained by the sample.
        sample_matrix = numpy.zeros((rows, self.grg.num_samples), dtype=numpy.int32)
        for k, sample_id in enumerate(sample_ids):
            assert sample_id < self.grg.num_samples
            sample_matrix[k, sample_id] = 1
        muts_for_samples = pygrgl.matmul(
            self.grg, sample_matrix, pygrgl.TraversalDirection.UP
        )

        # Hamming distance D(x, y) = |x| + |y| - 2*|x ^ y|, where "^" is intersection.
        return self.exact_hamming_dists(
            muts_for_samples,
            pygrgl.TraversalDirection.DOWN,
            emit_all_nodes=emit_all_nodes,
        )

    def exact_hamming_dists_by_mutation(
        self,
        mutation_ids: List[int],
        emit_all_nodes: bool = False,
    ) -> numpy.typing.NDArray:
        """
        Using exact computations, get the Hamming distances from the list of input mutation IDs (for Mutations in
        the GRG) to every other mutation in the GRG.

        :param mutation_ids: List of GRG Mutation IDs, each of which will be queried for distance to all other
            Mutations.
        :type mutation_ids: List[int]
        :param emit_all_nodes: Set to True to compute distances to every _node_ in the graph, not just every
            other Mutation. The output Matrix will have num_nodes columns when True.
        :type emit_all_nodes: bool
        :return: Matrix of distances, where each row corresponds to input Mutation IDs, and each column is the
            distance from the "other" Mutation ID. For example, if the 0th input is Mutation ID "m0", then the
            0th row of output is [D(m0, 0), D(m0, 1), ..., D(m0, M-1)] where M is the number of mutations.
        :rtype: numpy.ndarray
        """
        rows = len(mutation_ids)
        assert rows > 0
        # This computes a vector of 1's for every mutation that is contained by the sample.
        mut_matrix = numpy.zeros((rows, self.grg.num_mutations), dtype=numpy.int32)
        for k, mut_id in enumerate(mutation_ids):
            assert mut_id < self.grg.num_mutations
            mut_matrix[k, mut_id] = 1
        samples_for_muts = pygrgl.matmul(
            self.grg, mut_matrix, pygrgl.TraversalDirection.DOWN
        )

        # Hamming distance D(x, y) = |x| + |y| - 2*|x ^ y|, where "^" is intersection.
        return self.exact_hamming_dists(
            samples_for_muts,
            pygrgl.TraversalDirection.UP,
            emit_all_nodes=emit_all_nodes,
        )

    def fast_pairwise_hamming(
        self, node1: int, node2: int, direction: pygrgl.TraversalDirection
    ) -> int:
        """
        Compute the Hamming distance between a pair of samples or mutations (or arbitrary nodes in the graph,
        but that has a less well-defined "meaning").
        This calculation is extremely fast for a pair that are highly similar (low Hamming distance), as it
        shortcuts the graph traversal by making use of pygrgl.shared_frontier().

        Note: Ensure that node1 != node2 prior to calling.

        :param node1: The first node ID (e.g., sample ID or node associated with a mutation).
        :type node1: int
        :param node2: The second node ID (e.g., sample ID or node associated with a mutation).
        :type node2: int
        :param direction: The direction to use for distance calculation. pygrgl.TraversalDirection.UP means to
            compare the sets of Mutations shared by the nodes (distance is on differing Mutations) and
            pygrgl.TraversalDirection.DOWN means to compare sets of Samples.
        """
        frontier = pygrgl.shared_frontier(self.grg, direction, [node1, node2])
        # Hamming: |A| + |B| - 2*|A intersect B|
        if direction == pygrgl.TraversalDirection.UP:
            intersect = sum([self.muts_above[f] for f in frontier])
            A_size = self.muts_above[node1]
            B_size = self.muts_above[node2]
        else:
            assert direction == pygrgl.TraversalDirection.DOWN
            intersect = sum([self.samps_below[f] for f in frontier])
            A_size = self.samps_below[node1]
            B_size = self.samps_below[node2]
        return A_size + B_size - 2 * intersect
