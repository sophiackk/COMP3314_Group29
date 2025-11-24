import re
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_log_file(filename):
    """Parse log file and extract MSE/MAE results"""
    results = defaultdict(list)
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract all mse/mae lines
    pattern = r'mse:([\d.]+), mae:([\d.]+)'
    matches = re.findall(pattern, content)
    
    # Group by pred_len and seed
    pred_lens = [96, 192, 336, 720]
    seeds = [42, 123, 456]  
    
    idx = 0
    for seed in seeds:
        for pred_len in pred_lens:
            for iteration in range(3):  # 3 iterations per configuration
                if idx < len(matches):
                    mse, mae = matches[idx]
                    results[(pred_len, seed)].append({
                        'mse': float(mse),
                        'mae': float(mae),
                        'iteration': iteration + 1
                    })
                    idx += 1
    
    return results

def calculate_statistics(results):
    """Calculate mean and std for each pred_len across seeds"""
    stats = {}
    
    for pred_len in [96, 192, 336, 720]:
        mse_values = []
        mae_values = []
        
        for (pl, seed), runs in results.items():
            if pl == pred_len:
                mse_values.extend([run['mse'] for run in runs])
                mae_values.extend([run['mae'] for run in runs])
        
        if mse_values:
            stats[pred_len] = {
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values)
            }
    
    return stats

def save_summary_tables_to_csv(files_dict):
    """Save summary tables to CSV files"""
    
    all_stats = {}
    
    for variant, filename in files_dict.items():
        try:
            results = parse_log_file(filename)
            stats = calculate_statistics(results)
            all_stats[variant] = stats
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create MSE summary table
    mse_data = []
    for variant in all_stats.keys():
        row = {'Variant': variant}
        for pred_len in [96, 192, 336, 720]:
            if pred_len in all_stats[variant]:
                mse_mean = all_stats[variant][pred_len]['mse_mean']
                mse_std = all_stats[variant][pred_len]['mse_std']
                row[f'pred_{pred_len}_mse'] = f"{mse_mean:.4f}±{mse_std:.4f}"
            else:
                row[f'pred_{pred_len}_mse'] = "N/A"
        mse_data.append(row)
    
    mse_df = pd.DataFrame(mse_data)
    mse_df.to_csv('ablation_weather_mse.csv', index=False)
    print("Saved MSE summary to ablation_weather_mse.csv")
    
    # Create MAE summary table
    mae_data = []
    for variant in all_stats.keys():
        row = {'Variant': variant}
        for pred_len in [96, 192, 336, 720]:
            if pred_len in all_stats[variant]:
                mae_mean = all_stats[variant][pred_len]['mae_mean']
                mae_std = all_stats[variant][pred_len]['mae_std']
                row[f'pred_{pred_len}_mae'] = f"{mae_mean:.4f}±{mae_std:.4f}"
            else:
                row[f'pred_{pred_len}_mae'] = "N/A"
        mae_data.append(row)
    
    mae_df = pd.DataFrame(mae_data)
    mae_df.to_csv('ablation_weather_mae.csv', index=False)
    print("Saved MAE summary to ablation_weather_mae.csv")
    
    return mse_df, mae_df

# Main execution
if __name__ == "__main__":
    # Define log files for Weather iTransformer
    files_dict = {
        'Original': 'logs/weather/original.log',
        'w/o Z-Norm': 'logs/weather/no_znorm.log',
        'w/o SC': 'logs/weather/no_skip.log', 
        'VD-De': 'logs/weather/fuse_conv2d.log',
        'w/o SC & VD-De': 'logs/weather/no_skip_fuse_conv2d.log'
    }
    
    print("Processing Weather iTransformer results...")
    
    # Save summary tables
    mse_table, mae_table = save_summary_tables_to_csv(files_dict)
    
    print("\nProcessing completed!")
    print("Generated files:")
    print("- ablation_weather_mse.csv")
    print("- ablation_weather_mae.csv")
