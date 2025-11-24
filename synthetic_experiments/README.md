# Synthetic Data Experiments

Controlled synthetic datasets for evaluating transformer performance on time series forecasting.

## Structure

- `data_generation/` - Data generation scripts
- `datasets/` - 8 synthetic CSV files (2 γ × 4 α combinations)
- `analysis/` - Result extraction and visualization scripts
- `results/` - Experimental results and charts

## Data Generation

Generated 8 bivariate time series datasets with controlled properties:
- γ ∈ {0.5, 0.95}: autocorrelation strength
- α ∈ {0.0, 0.2, 0.4, 0.8}: inter-variate dependency
- 10,000 timesteps per dataset

Based on methodology in Li et al. (2024) Section 4.1.

Usage:
```bash
cd data_generation
python generate_synthetic.py
```

## Experiments

Tested PatchTST, Crossformer, and iTransformer on all datasets with prediction lengths 96 and 192.

Total: 48 experiments (3 models × 8 datasets × 2 prediction lengths)

## Key Findings

- Non-stationary data (γ=0.5) shows 25-30% performance degradation compared to stationary (γ=0.95)
- Stationarity has larger impact than architectural differences
- Performance exhibits non-monotonic relationship with α (optimal at α=0.4)

See full report for detailed analysis and results.
