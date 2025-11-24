"""
Extract experiment results from Time-Series-Library output folders
"""
import os
import re
import numpy as np
import pandas as pd

results_dir = './'
results = []

# Regex to parse folder names
pattern = r'long_term_forecast_(\w+)_gamma([\d.]+)_alpha([\d.]+)_pred(\d+)'

print("Extracting results...")

for folder in os.listdir(results_dir):
    if folder.startswith('long_term_forecast') and not folder.endswith('.py'):
        match = re.search(pattern, folder)
        if match:
            model, gamma, alpha, pred_len = match.groups()
            metrics_file = os.path.join(results_dir, folder, 'metrics.npy')
            
            try:
                metrics = np.load(metrics_file, allow_pickle=True)
                
                # Handle arrays vs scalars
                def to_scalar(val):
                    return float(val.item()) if isinstance(val, np.ndarray) else float(val)
                
                # MAE, MSE, RMSE, MAPE, MSPE
                mae, mse, rmse, mape, mspe = [to_scalar(metrics[i]) for i in range(5)]
                
                results.append({
                    'model': model,
                    'gamma': float(gamma),
                    'alpha': float(alpha),
                    'pred_length': int(pred_len),
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'mspe': mspe
                })
                
                print(f"{model:12s} γ={gamma} α={alpha} pl={pred_len} MAE={mae:.4f}")
                
            except Exception as e:
                print(f"Error: {folder} - {e}")

# Save results
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(['model', 'gamma', 'alpha', 'pred_length'])
    df.to_csv('all_results.csv', index=False)
    
    
    print("\nSummary:")
    print(f"  Models: {list(df['model'].unique())}")
    print(f"  Gamma: {list(df['gamma'].unique())}")
    print(f"  Alpha: {list(df['alpha'].unique())}")
    print(f"  Pred lengths: {list(df['pred_length'].unique())}")
    
    # Show sample
    print("\nSample (γ=0.95, pl=96):")
    sample = df[(df['pred_length'] == 96) & (df['gamma'] == 0.95)]
    print(sample[['model', 'alpha', 'mae', 'mse']].to_string(index=False))
else:
    print("\nNo results found!")