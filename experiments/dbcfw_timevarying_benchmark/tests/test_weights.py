from __future__ import annotations

import networkx as nx
import numpy as np

from dbcfw_bench.comm_graphs import GraphConfig, GraphSequence, metropolis_weights


def test_metropolis_weights_are_doubly_stochastic() -> None:
    graph = nx.path_graph(5)
    weights = metropolis_weights(graph, 5)
    np.testing.assert_allclose(weights.sum(axis=0), np.ones(5))
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(5))
    np.testing.assert_allclose(weights, weights.T)


def test_graph_sequence_is_reproducible() -> None:
    cfg = GraphConfig("erdos", edge_prob=0.6, seed=7)
    first = GraphSequence(6, cfg).next()[0]
    second = GraphSequence(6, cfg).next()[0]
    np.testing.assert_allclose(first, second)
