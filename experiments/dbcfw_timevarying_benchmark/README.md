# DFW vs DBCFW on Time-Varying Graphs

Runnable benchmarks for decentralized Frank-Wolfe with gradient tracking (DFW)
and decentralized block-coordinate Frank-Wolfe (DBCFW). The core problem is
`min f(x) = (1/N) sum_i f_i(x), x in D`; fair comparisons share objective seed,
graph seed, initialization, and the same generated `W_t` sequence.

## Commands

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m dbcfw_bench.cli run --method dbcfw --agents 20 --dim 1000 --blocks 50 --batch 5 --iters 500 --graph erdos --edge-prob 0.25 --seed 42 --out runs/single
python -m dbcfw_bench.cli run --objective structural_svm --method dbcfw --agents 6 --dim 60 --blocks 12 --batch 1 --iters 40 --reg 0.1 --lmo simplex --sequence-length 6 --label-count 3 --graph erdos --edge-prob 0.7 --seed 1207 --graph-seed 2207 --out runs/structural_svm_single
python -m dbcfw_bench.cli grid --config configs/default.yaml --out runs/grid
python -m dbcfw_bench.cli grid --config configs/structural_svm.yaml --out runs/structural_svm
python -m dbcfw_bench.cli plot --results runs/grid/results.csv --out runs/grid/plots
python -m dbcfw_bench.cli summarize --runs runs runs_lmo_compare --readme README.md
find . -name '*.py' -o -name '*.md' | xargs wc -l
```

## Metrics

Summary tables use accuracy when `mean_agent_accuracy` exists and otherwise use
point consensus `mean_i ||x_i - x_avg||`. `B/n` is the best DBCFW block fraction
for that metric. Step columns are the last logged per-iteration wall time.

Logged CSV fields include `objective_gap`, `mean_agent_accuracy`,
`consensus_error`, `gradient_tracker_disagreement`, `oracle_coordinates_per_iter`,
`total_oracle_coordinates`, `lambda2`, `boundary_activity`, `peak_memory_bytes`,
`gamma`, `wall_time_sec`, `method`, `batch`, `lmo`, seeds, and dimensions.

## Structural SVM Setup

`objective: structural_svm` runs a synthetic chain-labeling Structural SVM with
Hamming loss and exact loss-augmented Viterbi decoding. In this setup, `blocks`
is the number of local training examples per agent, not the number of coordinate
blocks in the model. The `dbcfw` method samples `batch` local examples per agent
and applies the primal-dual block-coordinate Frank-Wolfe line search from
Lacoste-Julien et al. (2013); `dfw` uses all local examples each communication
round. The runner keeps the decentralized time-varying communication graph and
logs `objective_gap` as the full Structural SVM Frank-Wolfe duality gap,
computed by an additional full oracle pass at logging points.

## Semi-Relaxed OT Setup

`python -m dbcfw_bench.cli ot` runs the semi-relaxed optimal transport problem
from https://arxiv.org/pdf/2103.05857:
`min_T <T,C> + 1/(2*lambda)||T 1 - a||^2` subject to nonnegative columns with
column sums `b`. Here each transport column is one FW block. The paper-style
runner compares decentralized full-column FW (`DFW`) against one-column
decentralized block-coordinate FW (`DBCFW`) while keeping the existing
time-varying decentralized graph and gradient tracking, with local cost matrices
averaged by consensus.

Example:

```bash
python -m dbcfw_bench.cli ot-paper --m 28 --n 28 --agents 8 --epochs 80 \
  --batch 1 --relaxations 0.02 0.04 0.08 0.16 0.32 \
  --convergence-relaxation 0.08 --transition-relaxations 0.02 0.32 \
  --cost-noise 0.03 --stepsize line_search --graph erdos --edge-prob 0.45 \
  --seed 202 --graph-seed 1202 --out runs_ot/paper_dbcfw_vs_dfw
```

Outputs include `paper_suite_results.csv`, transition CSVs, an LP reference, and
paper-analogue plots:

- Figure 1 analogue: relaxation sweep over objective, duality gap, marginal
  error, sparsity, matrix error, value error, and time.
- Figure 2 analogue: convergence panels against oracle epochs and wall time.
- Figure 4 analogue: column-duality-gap variance diagnostics.
- Figure 5 analogue: source/reference support and balanced OT LP reference.
- Figure 6/7 analogues: transition heatmaps plus objective/gradient components
  for two relaxation parameters.

For presentation-style images from an existing paper-suite run:

```bash
python -m dbcfw_bench.cli ot-gallery \
  --run-dir runs_ot/paper_dbcfw_vs_dfw
```

This writes a scorecard heatmap, duality-gap landscape, decentralized graph
diagnostics, and transport-geometry comparison under `<run-dir>/gallery/`.

For the color-transfer setup from the paper's Figure 9, build color palettes
from real images, solve semi-relaxed OT with `m=n=32`, and render transferred
images plus row-normalized transport heatmaps:

```bash
python -m dbcfw_bench.cli ot-color-transfer \
  --source rocket --target coffee --colors 32 --agents 8 --epochs 80 \
  --batch 1 --relaxation 0.08 --graph erdos --edge-prob 0.45 \
  --seed 202 --graph-seed 1202 --reference-epochs 1000 \
  --out runs_ot_color/figure9_rocket_to_coffee
```

The run saves source/target images, a high-budget centralized semi-relaxed
reference, balanced OT LP output, DFW/DBCFW transferred images, Figure-9-style
transition panels, heatmaps, and optimization diagnostics.

## Best Quality Comparison

Each row selects the setup with the best DBCFW quality inside a `family x LMO`
group and compares DFW on the same setup. For consensus, lower is better.


| Family              | LMO / set size                            | Best setup                        | Quality   | DFW       | Best DBCFW | B/n  |
| ------------------- | ----------------------------------------- | --------------------------------- | --------- | --------- | ---------- | ---- |
| Quadratic           | box; R=0.05; d=200; n=20; block=10        | `62_quadratic_q_d200_box_r005`    | consensus | 6.778e-05 | 3.336e-05  | 5%   |
| Quadratic           | l1_block; R=0.05; d=200; n=20; block=10   | `63_quadratic_q_d200_l1_r005`     | consensus | 2.774e-05 | 1.226e-05  | 5%   |
| Quadratic           | l2_block; R=0.05; d=1000; n=20; block=50  | `66_quadratic_q_d1000_l2_r005`    | consensus | 6.21e-15  | 6.71e-15   | 100% |
| MNIST               | box; R=0.5; d=7860; n=20; block=393       | `78_mnist_mnist_box_sparse_graph` | accuracy  | 0.9526    | 0.9594     | 25%  |
| MNIST               | l1_block; R=2.0; d=7860; n=20; block=393  | `14_mnist_mnist_l1_r2`            | accuracy  | 0.7096    | 0.7096     | 50%  |
| MNIST               | l2_block; R=2.0; d=7860; n=20; block=393  | `16_mnist_mnist_l2_r2`            | accuracy  | 0.935     | 0.935      | 100% |
| Fashion             | box; R=1.0; d=12740; n=20; block=637      | `21_fashion_fmlp_box_r1`          | accuracy  | 0.794     | 0.8876     | 5%   |
| Fashion             | l1_block; R=0.7; d=12740; n=20; block=637 | `82_fashion_fmlp_l1_r07`          | accuracy  | 0.1008    | 0.2476     | 50%  |
| Fashion             | l2_block; R=1.0; d=12740; n=20; block=637 | `25_fashion_fmlp_l2_r1`           | accuracy  | 0.476     | 0.6776     | 50%  |
| CIFAR               | box; R=1.0; d=7700; n=20; block=385       | `31_cifar_cifarlin_box_r1`        | accuracy  | 0.378     | 0.684      | 5%   |
| CIFAR               | l1_block; R=1.0; d=7700; n=20; block=385  | `33_cifar_cifarlin_l1_r1`         | accuracy  | 0.304     | 0.309      | 25%  |
| CIFAR               | l2_block; R=1.0; d=7700; n=20; block=385  | `35_cifar_cifarlin_l2_r1`         | accuracy  | 0.53      | 0.53       | 100% |
| Tabular             | box; R=5.0; d=120; n=20; block=6          | `41_tabular_mush_box_r5`          | accuracy  | 0.999     | 0.999      | 50%  |
| Tabular             | l1_block; R=5.0; d=120; n=20; block=6     | `44_tabular_mush_l1_r5`           | accuracy  | 0.999     | 0.999      | 50%  |
| Tabular             | l2_block; R=5.0; d=120; n=20; block=6     | `47_tabular_mush_l2_r5`           | accuracy  | 0.999     | 0.999      | 50%  |
| Text                | box; R=1.0; d=2000; n=20; block=100       | `51_text_sms_box_r1`              | accuracy  | 0.952     | 0.952      | 50%  |
| Text                | l1_block; R=1.0; d=2000; n=20; block=100  | `54_text_sms_l1_r1`               | accuracy  | 0.868     | 0.868      | 50%  |
| Text                | l2_block; R=0.1; d=2000; n=20; block=100  | `120_text_sms_l2_reg1e2`          | accuracy  | 0.88      | 0.88       | 50%  |
| Synthetic topic     | box; R=0.02; d=1000; n=20; block=50       | `01_topic_box_r002`               | accuracy  | 0.8128    | 0.8128     | 50%  |
| Synthetic topic     | l1_block; R=1.0; d=3000; n=30; block=100  | `07_topic_l1_r1_d3000`            | accuracy  | 0.7072    | 0.7078     | 100% |
| Synthetic topic     | l2_block; R=0.3; d=3000; n=30; block=100  | `10_topic_l2_sparse_d3000`        | accuracy  | 0.8406    | 0.8428     | 3%   |
| Synthetic sentiment | box; R=0.05; d=1000; n=20; block=50       | `12_sent_box_r005`                | accuracy  | 0.8222    | 0.8222     | 50%  |
| Synthetic sentiment | l1_block; R=0.5; d=2000; n=40; block=50   | `17_sent_l1_r05_d2000`            | accuracy  | 0.6683    | 0.67       | 25%  |
| Synthetic sentiment | l2_block; R=0.1; d=1000; n=20; block=50   | `19_sent_l2_r01`                  | accuracy  | 0.8233    | 0.8244     | 25%  |


## Constraint Sets

`box` uses `D=[-R,R]^d` with LMO `-R sign(g)`. `l1_block` and `l2_block`
apply the Frank-Wolfe LMO independently on each coordinate block. DFW is the
full-block case `B=n`; DBCFW samples `B` active blocks and keeps inactive blocks
at the mixed point.

## Communication Graphs

Each run uses a seed-controlled sequence of time-varying graphs:
`erdos_renyi_connected`, `random_geometric_connected`, or `pairwise_gossip`.
Edges are converted to doubly stochastic Metropolis weights and each row logs
`lambda2_t = ||W_t - J||_2`.

## Generated Outputs

The 120-run hyper/constraint sweep is summarized in `runs_hyper_sweep/summary.md`.
Per-family tables are `runs_hyper_sweep/summary_<family>.md`.
Per-setup plots live under `artifacts/hyper_sweep/<family>/<hypers>/`.

The 20-run synthetic NLP sweep is summarized in `runs_nlp_sweep/summary.md`.
NLP plots live under `artifacts/nlp_sweep/<family>/<hypers>/`.
