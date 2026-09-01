# Paper OCR Structural SVM Experiments

The ICML 2013 OCR setup uses Taskar OCR with the `ocr2` split: folds other than
0 for training and fold 0 for test. The original BCFWstruct demo uses
loss-augmented Viterbi, normalized Hamming loss, `lambda in {0.01, 0.001, 1/n}`,
line-search for FW/BCFW, weighted averaging for BCFW-wavg, and effective passes
on the x-axis.

Reduced validation plus decentralized DBCFW/DFW:

```bash
python -m dbcfw_bench.paper_ocr \
  --mode central decentralized \
  --methods bcfw fw \
  --data-dir data \
  --out runs_paper_ocr/ocr_reduced_all_lambda \
  --lambdas 0.01 0.001 1/n \
  --passes 10 \
  --log-every 2 \
  --agents 7 \
  --blocks 893 \
  --batches 1 10 89 \
  --decentralized-iters 20 \
  --decentralized-log-every 5 \
  --edge-prob 0.7 \
  --seed 1 \
  --graph-seed 2207
```

Full Figure 1/3-style OCR validation:

```bash
python -m dbcfw_bench.paper_ocr \
  --mode central \
  --methods bcfw fw \
  --data-dir data \
  --out runs_paper_ocr/ocr_full_140_passes \
  --lambdas 0.01 0.001 1/n \
  --passes 140 \
  --log-every 5 \
  --seed 1
```

Full decentralized OCR run on the same train set split across 7 agents:

```bash
python -m dbcfw_bench.paper_ocr \
  --mode decentralized \
  --data-dir data \
  --out runs_paper_ocr/ocr_decentralized_140_iters \
  --lambdas 0.01 0.001 1/n \
  --agents 7 \
  --blocks 893 \
  --batches 1 10 89 \
  --decentralized-iters 140 \
  --decentralized-log-every 5 \
  --edge-prob 0.7 \
  --seed 1 \
  --graph-seed 2207
```

3D paper-style visualizations from saved CSV files:

```bash
python -m dbcfw_bench.paper_ocr_3d \
  --central runs_paper_ocr/ocr_full_140_passes/results.csv \
  --decentralized runs_paper_ocr/ocr_decentralized_140_iters/results.csv \
  --out runs_paper_ocr/plots_3d
```
