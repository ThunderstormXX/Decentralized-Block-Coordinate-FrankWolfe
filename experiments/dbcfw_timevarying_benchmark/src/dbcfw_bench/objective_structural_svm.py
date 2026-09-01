from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dbcfw_bench.objective import ReferenceSolution


@dataclass
class StructuralSequenceSVMProblem:
    x_parts: list[list[np.ndarray] | np.ndarray]
    y_parts: list[list[np.ndarray] | np.ndarray]
    reg: float
    classes: int
    position_bias: bool = False
    test_x: list[np.ndarray] | None = None
    test_y: list[np.ndarray] | None = None
    true_feature_parts: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.true_feature_parts = [
            np.stack([self.feature_map(x_i[j], y_i[j]) for j in range(len(y_i))])
            for x_i, y_i in zip(self.x_parts, self.y_parts)
        ]

    @property
    def agents(self) -> int:
        return len(self.x_parts)

    @property
    def block_count(self) -> int:
        return len(self.x_parts[0])

    @property
    def dim(self) -> int:
        token_dim = int(self.x_parts[0][0].shape[1])
        bias_dim = 2 * self.classes if self.position_bias else 0
        return token_dim * self.classes + bias_dim + self.classes * self.classes

    @property
    def total_examples(self) -> int:
        return int(sum(len(part) for part in self.y_parts))

    def objective(self, w: np.ndarray) -> float:
        return 0.5 * self.reg * float(w @ w) + self.average_hinge(w)

    def average_hinge(self, w: np.ndarray) -> float:
        total = 0.0
        for agent in range(self.agents):
            for block in range(len(self.y_parts[agent])):
                _, _, _, hinge = self.loss_augmented_decode(agent, block, w)
                total += hinge
        return total / self.total_examples

    def oracle_vertex(
        self, agent: int, block: int, w: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        _, loss, psi, hinge = self.loss_augmented_decode(agent, block, w)
        scale = 1.0 / (self.reg * self.total_examples)
        return scale * psi, loss / self.total_examples, psi, hinge

    def loss_augmented_decode(
        self, agent: int, block: int, w: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        x_seq = self.x_parts[agent][block]
        y_true = self.y_parts[agent][block]
        y_hat = self._viterbi_loss_augmented(x_seq, y_true, w)
        phi_true = self.true_feature_parts[agent][block]
        phi_hat = self.feature_map(x_seq, y_hat)
        loss = self.sequence_loss(y_true, y_hat)
        psi = phi_true - phi_hat
        hinge = loss - float(w @ psi)
        return y_hat, loss, psi, max(hinge, 0.0)

    def full_oracle(self, w: np.ndarray) -> tuple[np.ndarray, float, int]:
        ws = np.zeros(self.dim, dtype=float)
        ell = 0.0
        calls = 0
        for agent in range(self.agents):
            for block in range(len(self.y_parts[agent])):
                vertex, loss_value, _, _ = self.oracle_vertex(agent, block, w)
                ws += vertex
                ell += loss_value
                calls += 1
        return ws, ell, calls

    def duality_gap(self, w: np.ndarray, ell: float) -> tuple[float, int]:
        ws, ell_s, calls = self.full_oracle(w)
        gap = self.reg * float((w - ws) @ w) - ell + ell_s
        return max(gap, 0.0), calls

    def mean_agent_accuracy(self, points: np.ndarray) -> float:
        values = []
        for agent in range(self.agents):
            correct = 0
            total = 0
            for block in range(len(self.y_parts[agent])):
                pred = self.decode(self.x_parts[agent][block], points[agent])
                target = self.y_parts[agent][block]
                correct += int(np.sum(pred == target))
                total += len(target)
            values.append(correct / max(total, 1))
        return float(np.mean(values))

    def average_sequence_loss(self, x_data: list[np.ndarray], y_data: list[np.ndarray], w: np.ndarray) -> float:
        losses = [self.sequence_loss(y, self.decode(x, w)) for x, y in zip(x_data, y_data)]
        return float(np.mean(losses)) if losses else float("nan")

    def test_error(self, w: np.ndarray) -> float:
        if self.test_x is None or self.test_y is None:
            return float("nan")
        return self.average_sequence_loss(self.test_x, self.test_y, w)

    def decode(self, x_seq: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self._viterbi(x_seq, np.zeros((len(x_seq), self.classes)), w)

    def feature_map(self, x_seq: np.ndarray, y_seq: np.ndarray) -> np.ndarray:
        token_dim = x_seq.shape[1]
        emissions = np.zeros((token_dim, self.classes), dtype=float)
        for token, label in enumerate(y_seq):
            emissions[:, int(label)] += x_seq[token]
        chunks = [emissions.ravel()]
        if self.position_bias:
            first = np.zeros(self.classes, dtype=float)
            last = np.zeros(self.classes, dtype=float)
            first[int(y_seq[0])] = 1.0
            last[int(y_seq[-1])] = 1.0
            chunks.extend([first, last])
        transitions = np.zeros((self.classes, self.classes), dtype=float)
        for token in range(1, len(y_seq)):
            transitions[int(y_seq[token - 1]), int(y_seq[token])] += 1.0
        chunks.append(transitions.ravel())
        return np.concatenate(chunks)

    def sequence_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true != y_pred))

    def solve_reference(self, maxiter: int = 300) -> ReferenceSolution:
        return ReferenceSolution(
            np.zeros(self.dim),
            0.0,
            True,
            "structural_svm_duality_gap_logged_as_objective_gap",
        )

    def _viterbi_loss_augmented(
        self, x_seq: np.ndarray, y_true: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        loss_scores = (np.arange(self.classes)[None, :] != y_true[:, None]) / len(y_true)
        return self._viterbi(x_seq, loss_scores.astype(float), w)

    def _viterbi(self, x_seq: np.ndarray, extra_scores: np.ndarray, w: np.ndarray) -> np.ndarray:
        token_dim = x_seq.shape[1]
        emission_w = w[: token_dim * self.classes].reshape(token_dim, self.classes)
        offset = token_dim * self.classes
        emission_scores = x_seq @ emission_w + extra_scores
        if self.position_bias:
            emission_scores[0] += w[offset : offset + self.classes]
            offset += self.classes
            emission_scores[-1] += w[offset : offset + self.classes]
            offset += self.classes
        transition_w = w[offset:].reshape(self.classes, self.classes)
        dp = np.zeros_like(emission_scores)
        back = np.zeros((len(x_seq), self.classes), dtype=int)
        dp[0] = emission_scores[0]
        for token in range(1, len(x_seq)):
            scores = dp[token - 1][:, None] + transition_w + emission_scores[token][None, :]
            back[token] = np.argmax(scores, axis=0)
            dp[token] = np.max(scores, axis=0)
        labels = np.zeros(len(x_seq), dtype=int)
        labels[-1] = int(np.argmax(dp[-1]))
        for token in range(len(x_seq) - 1, 0, -1):
            labels[token - 1] = back[token, labels[token]]
        return labels
