# summer-2026-dl-alpha-mining-via-neural-network

2026 Summer Erdos Institute Deep Learning Bootcamp Project Repository.

## M2-C Sector-GCN

The repository includes a dependency-free, two-layer NumPy GCN matching the
proposal's M2-C ablation. Each trading date is a 500-node graph; stocks with
the same `sector_index` are connected, one self-loop is added per node, and
the model is trained with daily-equal five-class cross-entropy.

Run one chronological fold first:

```bash
python script/fit_m2_gcn.py --fold 0
```

Run every prepared walk-forward fold:

```bash
python script/fit_m2_gcn.py
```

Publish its OOS probabilities and build model-specific portfolios:

```bash
python script/export_test_probabilities.py \
  --source-dir results/models/M2-C \
  --output-dir results/probabilities/M2-C
python script/build_ranked_portfolios.py \
  --input-dir results/probabilities/M2-C \
  --score-dir results/ranking_scores/M2-C \
  --portfolio-dir results/portfolios/M2-C
python script/evaluate_prediction_metrics.py \
  --probability-dir results/probabilities/M2-C \
  --score-dir results/ranking_scores/M2-C \
  --output-dir results/prediction_metrics/M2-C
python script/evaluate_portfolio_metrics.py \
  --portfolio-dir results/portfolios/M2-C \
  --output-dir results/portfolio_metrics/M2-C
```

The default configuration is `configs/training/M2-C.json`, the model code is
in `src/alpha_mining_neural_network/gcn.py`, and the chronological training
workflow is in `src/alpha_mining_neural_network/m2_gcn.py`. Model-ready input
is expected under `data/ready_for_use`, following the same tensor and split
contract as M0.

M2-C intentionally follows the proposal's complete same-sector graph. Its
normalized clique averages all member features, so stocks sharing a sector
receive identical predictions and cannot be ranked within that sector. This
is retained as a faithful ablation baseline; a practical follow-up should use
a separately named residual GCN or a sparse correlation graph.
