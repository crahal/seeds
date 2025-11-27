import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_stability(n_results, prop):
    """
    Visualizes seed variability results for the GRF.
    """
    files = {
        'impute': (f"brand_ate_stability_impute_{n_results}_{prop}.csv", "skyblue", "Variation due to Imputation"),
        'sample': (f"brand_ate_stability_sample_{n_results}_{prop}.csv", "orange", "Variation due to Sampling"),
        'model': (f"brand_ate_stability_model_{n_results}_{prop}.csv", "green", "Variation due to Model Training"),
        'all': (f"brand_ate_stability_all_{n_results}_{prop}.csv", "purple", "Overall Variation")
    }
    
    # Set global style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ATE Stability Analysis: Decomposition of Variance', fontsize=16)
    
    # Map keys to axes
    ax_map = {
        'impute': axes[0, 0],
        'sample': axes[0, 1],
        'model': axes[1, 0],
        'all': axes[1, 1]
    }

    for key, (filepath, color, title) in files.items():
        ax = ax_map[key]
        
        if not os.path.exists(filepath):
            # Check src/ prefix if running from root
            if os.path.exists(os.path.join("src", filepath)):
                filepath = os.path.join("src", filepath)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if 'ate' in df.columns and len(df) > 1:
                    mean_val = df['ate'].mean()
                    var_val = df['ate'].var()
                    std_val = df['ate'].std()
                    if std_val < 1e-9:
                        sns.histplot(data=df, x='ate', ax=ax, color=color, bins=10)
                        ax.set_title(f'{title}\n(No Variation: Std Dev ≈ 0)')
                        ax.set_xlim(-0.4, 0.05)
                        ax.axvline(x=-0.176, color='black', linestyle='--', linewidth=1.5, label='original estimate')
                        ax.legend()
                    else:
                        sns.kdeplot(data=df, x='ate', ax=ax, fill=True, color=color)
                        ax.set_title(f'{title}\n(N={len(df)})')
                    ax.set_xlabel('ATE')
                    ax.set_xlim(-0.4, 0.05)
                    ax.axvline(x=-0.176, color='black', linestyle='--', linewidth=1.5, label='original estimate')
                    ax.legend()
                    # Add mean and variance text
                    stats_text = f'Mean: {mean_val:.4f}\nVariance: {var_val:.6f}'
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=9)
                else:
                    ax.text(0.5, 0.5, f"Not enough data in\n{os.path.basename(filepath)}", ha='center', va='center')
                    ax.set_title(title)
                    ax.set_xlim(-0.4, 0.05)
                    ax.axvline(x=-0.176, color='black', linestyle='--', linewidth=1.5, label='original estimate')
                    ax.legend()
            except Exception as e:
                ax.text(0.5, 0.5, f"Error reading file:\n{e}", ha='center', va='center')
                ax.set_title(title)
                ax.set_xlim(-0.4, 0.05)
                ax.axvline(x=-0.176, color='black', linestyle='--', linewidth=1.5, label='original estimate')
                ax.legend()
        else:
            ax.text(0.5, 0.5, f"File not found:\n{filepath}", ha='center', va='center')
            ax.set_title(title)
            ax.set_xlim(-0.4, 0.05)
            ax.axvline(x=-0.176, color='black', linestyle='--', linewidth=1.5, label='original estimate')
            ax.legend()

    plt.tight_layout()
    
    output_plot = f"ate_stability_density_combined_{n_results}_{prop}.png"
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    visualize_stability(n_results=1000, prop=0.3)

