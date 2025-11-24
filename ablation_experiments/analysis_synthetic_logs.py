import re
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_synthetic_log_file(filename):
    """Parse synthetic log file using the fixed order from grep output"""
    results = defaultdict(lambda: defaultdict(list))
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract all mse/mae lines
    pattern = r'mse:([\d.]+), mae:([\d.]+)'
    matches = re.findall(pattern, content)
    
    datasets = ['gamma0.5_alpha0.0', 'gamma0.95_alpha0.0', 'gamma0.5_alpha0.8', 'gamma0.95_alpha0.8']
    seeds = [42, 123, 456]
    pred_lens = [96, 192, 336, 720]
    
    idx = 0
    for dataset in datasets:
        for seed in seeds:
            for pred_len in pred_lens:
                for iteration in range(3):  # 3 iterations per configuration
                    if idx < len(matches):
                        mse, mae = matches[idx]
                        results[dataset][(pred_len, seed)].append({
                            'mse': float(mse),
                            'mae': float(mae),
                            'iteration': iteration + 1
                        })
                        idx += 1
    
    return results

def calculate_synthetic_statistics(results):
    """Calculate mean and std for each dataset and pred_len"""
    stats = {}
    
    for dataset, dataset_results in results.items():
        stats[dataset] = {}
        for pred_len in [96, 192, 336, 720]:
            mse_values = []
            mae_values = []
            
            for (pl, seed), runs in dataset_results.items():
                if pl == pred_len:
                    mse_values.extend([run['mse'] for run in runs])
                    mae_values.extend([run['mae'] for run in runs])
            
            if mse_values:
                stats[dataset][pred_len] = {
                    'mse_mean': np.mean(mse_values),
                    'mse_std': np.std(mse_values),
                    'mae_mean': np.mean(mae_values),
                    'mae_std': np.std(mae_values)
                }
    
    return stats

def save_synthetic_summary_tables(files_dict):
    """Save summary tables for each synthetic dataset to CSV files"""
    
    all_stats = {}
    
    for variant, filename in files_dict.items():
        try:
            print(f"Processing {filename}...")
            results = parse_synthetic_log_file(filename)
            stats = calculate_synthetic_statistics(results)
            all_stats[variant] = stats
            print(f"  Processed {len(results)} datasets")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Define the 4 synthetic datasets
    target_datasets = ['gamma0.5_alpha0.0', 'gamma0.5_alpha0.8', 'gamma0.95_alpha0.0', 'gamma0.95_alpha0.8']
    
    # Create separate CSV files for each dataset
    for dataset in target_datasets:
        # MSE table for this dataset
        mse_data = []
        for variant in all_stats.keys():
            if dataset in all_stats[variant]:
                row = {'Variant': variant}
                for pred_len in [96, 192, 336, 720]:
                    if pred_len in all_stats[variant][dataset]:
                        mse_mean = all_stats[variant][dataset][pred_len]['mse_mean']
                        mse_std = all_stats[variant][dataset][pred_len]['mse_std']
                        row[f'pred_{pred_len}_mse'] = f"{mse_mean:.4f}±{mse_std:.4f}"
                    else:
                        row[f'pred_{pred_len}_mse'] = "N/A"
                mse_data.append(row)
        
        if mse_data:
            mse_df = pd.DataFrame(mse_data)
            mse_filename = f'ablation_synthetic_{dataset}_mse.csv'
            mse_df.to_csv(mse_filename, index=False)
            print(f"Saved MSE summary to {mse_filename}")
        
        # MAE table for this dataset
        mae_data = []
        for variant in all_stats.keys():
            if dataset in all_stats[variant]:
                row = {'Variant': variant}
                for pred_len in [96, 192, 336, 720]:
                    if pred_len in all_stats[variant][dataset]:
                        mae_mean = all_stats[variant][dataset][pred_len]['mae_mean']
                        mae_std = all_stats[variant][dataset][pred_len]['mae_std']
                        row[f'pred_{pred_len}_mae'] = f"{mae_mean:.4f}±{mae_std:.4f}"
                    else:
                        row[f'pred_{pred_len}_mae'] = "N/A"
                mae_data.append(row)
        
        if mae_data:
            mae_df = pd.DataFrame(mae_data)
            mae_filename = f'ablation_synthetic_{dataset}_mae.csv' 
            mae_df.to_csv(mae_filename, index=False)
            print(f"Saved MAE summary to {mae_filename}")

# Main execution
if __name__ == "__main__":
    # Define log files for Synthetic iTransformer
    files_dict = {
        'Original': 'logs/synthetic/original.log',
        'w/o Z-Norm': 'logs/synthetic/no_znorm.log',
        'w/o SC': 'logs/synthetic/no_skip.log', 
        'VD-De': 'logs/synthetic/fuse_conv2d.log',
        'w/o SC & VD-De': 'logs/synthetic/no_skip_fuse_conv2d.log'
    }
    
    print("Processing Synthetic iTransformer results...")
    
    # Save summary tables for each synthetic dataset
    save_synthetic_summary_tables(files_dict)
    
    print("\nProcessing completed!")
    print("Generated 8 CSV files:")
    print("1. ablation_synthetic_gamma0.5_alpha0.0_mse.csv")
    print("2. ablation_synthetic_gamma0.5_alpha0.0_mae.csv")
    print("3. ablation_synthetic_gamma0.5_alpha0.8_mse.csv")
    print("4. ablation_synthetic_gamma0.5_alpha0.8_mae.csv") 
    print("5. ablation_synthetic_gamma0.95_alpha0.0_mse.csv")
    print("6. ablation_synthetic_gamma0.95_alpha0.0_mae.csv")
    print("7. ablation_synthetic_gamma0.95_alpha0.8_mse.csv")
    print("8. ablation_synthetic_gamma0.95_alpha0.8_mae.csv")
