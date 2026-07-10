# M0 OOS figures

Generate the proposal-defined M0 diagnostic figures from the saved prediction
and portfolio metrics:

```bash
python models/M0/plot_oos_results.py
```

The script uses the headline 10 bps one-way transaction-cost scenario recorded
in `results/portfolio_metrics/portfolio_metrics.json`. It writes individual PNG
and PDF figures plus a combined dashboard to `results/figures/`.
