from __future__ import annotations

import numpy as np

from dbcfw_bench.config import RunConfig
from dbcfw_bench.data_cifar import make_cifar_problem
from dbcfw_bench.data_fashion import make_fashion_cnn_problem, make_fashion_mlp_problem
from dbcfw_bench.data_mnist import make_mnist_problem
from dbcfw_bench.data_mushrooms import make_mushroom_problem
from dbcfw_bench.data_nlp import make_synthetic_nlp_problem
from dbcfw_bench.data_ocr import make_ocr_structural_svm_problem
from dbcfw_bench.data_structural_svm import make_structural_sequence_svm_problem
from dbcfw_bench.data_text import make_sms_problem
from dbcfw_bench.objective_flow_matching import make_euclidean_flow_matching_problem
from dbcfw_bench.objective import QuadraticProblem


def make_problem(config: RunConfig):
    if config.objective == "mushrooms_logreg":
        return make_mushroom_problem(config)
    if config.objective == "mnist_multiclass_logreg":
        return make_mnist_problem(config)
    if config.objective == "mnist_multiclass_logreg_noniid":
        return make_mnist_problem(config, non_iid=True)
    if config.objective == "fashion_mnist_mlp":
        return make_fashion_mlp_problem(config)
    if config.objective == "fashion_mnist_mlp_noniid":
        return make_fashion_mlp_problem(config, non_iid=True)
    if config.objective == "fashion_mnist_cnn":
        return make_fashion_cnn_problem(config)
    if config.objective == "fashion_mnist_cnn_noniid":
        return make_fashion_cnn_problem(config, non_iid=True)
    if config.objective == "cifar10_linear":
        return make_cifar_problem(config, "linear")
    if config.objective == "cifar10_shallow_cnn":
        return make_cifar_problem(config, "cnn")
    if config.objective == "sms_spam_logreg":
        return make_sms_problem(config)
    if config.objective == "synthetic_topic_logreg":
        return make_synthetic_nlp_problem(config, "topic")
    if config.objective == "synthetic_sentiment_logreg":
        return make_synthetic_nlp_problem(config, "sentiment")
    if config.objective in {"structural_svm", "structural_sequence_svm"}:
        return make_structural_sequence_svm_problem(config)
    if config.objective == "ocr_structural_svm":
        return make_ocr_structural_svm_problem(config)
    if config.objective == "euclidean_flow_matching":
        return make_euclidean_flow_matching_problem(config)
    rng = np.random.default_rng(config.seed)
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    x_true = rng.uniform(-0.7 * config.box_radius, 0.7 * config.box_radius, config.dim)
    scale = 1.0 / np.sqrt(config.dim)
    for _ in range(config.agents):
        a_i = rng.normal(0.0, scale, (config.samples_per_agent, config.dim))
        noise = 0.05 * rng.normal(size=config.samples_per_agent)
        b_i = a_i @ x_true + noise
        a_parts.append(a_i.astype(float, copy=False))
        b_parts.append(b_i.astype(float, copy=False))
    return QuadraticProblem(a_parts, b_parts, config.reg, config.box_radius)
