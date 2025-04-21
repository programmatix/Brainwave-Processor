import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_correlations(df, var1, var2):
    pearson_corr, pearson_p = stats.pearsonr(df[var1], df[var2])
    spearman_corr, spearman_p = stats.spearmanr(df[var1], df[var2])
    kendall_corr, kendall_p = stats.kendalltau(df[var1], df[var2])
    
    results = {
        'Pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
        'Spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
        'Kendall': {'correlation': kendall_corr, 'p_value': kendall_p}
    }
    
    return results

def print_correlation_results(results):
    for method, values in results.items():
        corr = values['correlation']
        p_val = values['p_value']
        print(f"{method} correlation: {corr:.4f}, p-value: {p_val:.4f}")

def visualize_scatter(df, var1, var2, correlation_results=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[var1], y=df[var2])
    plt.xlabel(var1)
    plt.ylabel(var2)
    
    if correlation_results:
        title = f"Scatter plot of {var1} vs {var2}\n"
        for method, values in correlation_results.items():
            corr = values['correlation']
            title += f"{method}: {corr:.4f}  "
        plt.title(title)
    else:
        plt.title(f"Scatter plot of {var1} vs {var2}")
    
    plt.tight_layout()
    plt.show()

def visualize_joint_plot(df, var1, var2, kind='reg'):
    g = sns.jointplot(x=df[var1], y=df[var2], kind=kind)
    g.fig.suptitle(f"Joint plot of {var1} vs {var2}", y=1.02)
    plt.tight_layout()
    plt.show()

def visualize_correlation_heatmap(df, variables=None):
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[variables].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def analyze_two_variables(df, var1, var2):
    results = calculate_correlations(df, var1, var2)
    print_correlation_results(results)
    
    visualize_scatter(df, var1, var2, results)
    visualize_joint_plot(df, var1, var2)
    
    return results

# Example usage:
# df = pd.read_csv('your_data.csv')
# analyze_two_variables(df, 'variable1', 'variable2')
