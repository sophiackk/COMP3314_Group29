"""
Synthetic time series generation for transformer experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_data(gamma, alpha, n_samples=10000, k=24, seed=None):
    """Generate synthetic time series with controlled dependencies"""
    
    if seed is not None:
        np.random.seed(seed)
    
    # Exponential weights for lookback
    tau = np.arange(1, k+1)
    omega = np.exp(-tau / k)
    omega = omega / omega.sum()
    
    print(f"Generating γ={gamma}, α={alpha}")
    
    a = np.zeros(n_samples)
    b = np.zeros(n_samples)
    
    # Initialize with random noise
    a[:k] = np.random.randn(k)
    b[:k] = np.random.randn(k)
    
    # Generate series
    for t in range(k, n_samples):
        past_a = a[t-k:t][::-1]
        past_b = b[t-k:t][::-1]
        
        # Variable a: autocorrelated
        trend_a = gamma * np.sum(omega * past_a)
        a[t] = trend_a + np.sqrt(1 - gamma**2) * np.random.randn()
        
        # Variable b: depends on both a and its own history
        cross_term = alpha * a[t]
        trend_b = gamma * np.sum(omega * past_b)
        self_term = np.sqrt(1 - alpha**2) * (trend_b + np.sqrt(1 - gamma**2) * np.random.randn())
        b[t] = cross_term + self_term
    
    df = pd.DataFrame({'a': a, 'b': b})
    
    # Quick check
    corr = np.corrcoef(a, b)[0, 1]
    print(f"  Correlation: {corr:.3f} (target: {alpha})")
    
    return df


def plot_data(data, gamma, alpha, save_path=None):
    """Visualization of generated data"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    axes[0].plot(data['a'][:500], label='a', alpha=0.8, color='blue')
    axes[0].plot(data['b'][:500], label='b', alpha=0.8, color='red')
    axes[0].set_title(f'γ={gamma}, α={alpha}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(data['a'], data['b'], alpha=0.1, s=1)
    corr = np.corrcoef(data['a'], data['b'])[0, 1]
    axes[1].set_title(f'Correlation: {corr:.3f}')
    axes[1].set_xlabel('Variable a')
    axes[1].set_ylabel('Variable b')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Test generation
    print("Testing synthetic data generation...")
    
    # High autocorrelation, no cross-dependency
    data1 = generate_synthetic_data(gamma=0.95, alpha=0.0, n_samples=5000, seed=42)
    plot_data(data1, 0.95, 0.0, 'test_g095_a00.png')
    
    # High autocorrelation, strong cross-dependency  
    data2 = generate_synthetic_data(gamma=0.95, alpha=0.8, n_samples=5000, seed=100)
    plot_data(data2, 0.95, 0.8, 'test_g095_a08.png')
    
    # Low autocorrelation cases
    data3 = generate_synthetic_data(gamma=0.5, alpha=0.0, n_samples=5000, seed=200)
    data4 = generate_synthetic_data(gamma=0.5, alpha=0.8, n_samples=5000, seed=300)
    