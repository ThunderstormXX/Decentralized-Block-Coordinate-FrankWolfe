from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dbcfw_bench.config import graph_name


@dataclass
class GraphConfig:
    model: str = "erdos"
    edge_prob: float = 0.25
    geometric_radius: float = 0.4
    seed: int = 0


def metropolis_weights(graph: nx.Graph, n_agents: int) -> np.ndarray:
    weights = np.zeros((n_agents, n_agents), dtype=float)
    degrees = dict(graph.degree())
    for i, j in graph.edges():
        value = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        weights[i, j] = value
        weights[j, i] = value
    row_sums = weights.sum(axis=1)
    weights[np.arange(n_agents), np.arange(n_agents)] = 1.0 - row_sums
    return weights


def spectral_contraction(weights: np.ndarray) -> float:
    n_agents = weights.shape[0]
    projector = np.ones((n_agents, n_agents), dtype=float) / n_agents
    return float(np.linalg.norm(weights - projector, ord=2))


class GraphSequence:
    def __init__(self, n_agents: int, config: GraphConfig):
        self.n_agents = n_agents
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def next(self) -> tuple[np.ndarray, float]:
        graph = sample_graph(self.n_agents, self.config, self.rng)
        weights = metropolis_weights(graph, self.n_agents)
        return weights, spectral_contraction(weights)


def sample_graph(n_agents: int, config: GraphConfig, rng: np.random.Generator) -> nx.Graph:
    model = graph_name(config.model)
    if model == "erdos_renyi_connected":
        return _erdos_connected(n_agents, config.edge_prob, rng)
    if model == "random_geometric_connected":
        return _geometric_connected(n_agents, config.geometric_radius, rng)
    if model == "pairwise_gossip":
        return _random_matching(n_agents, rng)
    raise ValueError(f"unknown graph model: {config.model}")


def _erdos_connected(n_agents: int, edge_prob: float, rng: np.random.Generator) -> nx.Graph:
    prob = min(max(edge_prob, 1e-3), 1.0)
    for _ in range(500):
        seed = int(rng.integers(0, 2**32 - 1))
        graph = nx.erdos_renyi_graph(n_agents, prob, seed=seed)
        if nx.is_connected(graph):
            return graph
    return nx.complete_graph(n_agents)


def _geometric_connected(n_agents: int, radius: float, rng: np.random.Generator) -> nx.Graph:
    rad = max(radius, 1e-3)
    for _ in range(500):
        seed = int(rng.integers(0, 2**32 - 1))
        graph = nx.random_geometric_graph(n_agents, rad, seed=seed)
        if nx.is_connected(graph):
            return graph
    return nx.complete_graph(n_agents)


def _random_matching(n_agents: int, rng: np.random.Generator) -> nx.Graph:
    order = rng.permutation(n_agents)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_agents))
    for pos in range(0, n_agents - 1, 2):
        graph.add_edge(int(order[pos]), int(order[pos + 1]))
    return graph
