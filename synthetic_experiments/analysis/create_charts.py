"""
Create visualization charts from experiment results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('all_results.csv')

sns.set_style("whitegrid")
sns.set_palette("husl")

print("Creating charts...")

# Chart 1: Bar chart showing alpha effect
df_plot = df[(df['gamma'] == 0.95) & (df['pred_length'] == 96)]
df_pivot = df_plot.pivot(index='alpha', columns='model', values='mae')

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_pivot.index))
width = 0.25
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, (model, color) in enumerate(zip(df_pivot.columns, colors)):
    ax.bar(x + i*width, df_pivot[model], width, label=model, color=color, alpha=0.8)

ax.set_xlabel('α (Inter-variate Dependency)', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE', fontsize=13, fontweight='bold')
ax.set_title('Model Performance vs Data Properties\n(γ=0.95, Prediction Length=96)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels([f'{a:.1f}' for a in df_pivot.index])
ax.legend(title='Model')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('chart1_alpha_effect.png', dpi=300, bbox_inches='tight')
print("chart1_alpha_effect.png")
plt.close()

# Chart 2: Line plots comparing gamma values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# High autocorrelation (gamma=0.95)
df_high = df[(df['gamma'] == 0.95) & (df['pred_length'] == 96)]
for model in ['PatchTST', 'iTransformer', 'Crossformer']:
    data = df_high[df_high['model'] == model].sort_values('alpha')
    ax1.plot(data['alpha'], data['mae'], marker='o', linewidth=2.5, 
             label=model, markersize=8)

ax1.set_xlabel('α (Inter-variate Dependency)', fontsize=11, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax1.set_title('Stationary (γ=0.95)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim([0.55, 0.85])

# Low autocorrelation (gamma=0.5)
df_low = df[(df['gamma'] == 0.5) & (df['pred_length'] == 96)]
for model in ['PatchTST', 'iTransformer', 'Crossformer']:
    data = df_low[df_low['model'] == model].sort_values('alpha')
    ax2.plot(data['alpha'], data['mae'], marker='o', linewidth=2.5, 
             label=model, markersize=8)

ax2.set_xlabel('α (Inter-variate Dependency)', fontsize=11, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax2.set_title('Non-Stationary (γ=0.5)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim([0.75, 0.85])

plt.tight_layout()
plt.savefig('chart2_gamma_comparison.png', dpi=300, bbox_inches='tight')
print("chart2_gamma_comparison.png")
plt.close()

print("\nDone! Charts saved:")
print("chart1_alpha_effect.png")
print("chart2_gamma_comparison.png")