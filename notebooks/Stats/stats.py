import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress_grid

def advanced_visualizations(X, y, model, title, top_n=10):
    """Generate advanced visualizations for regression models"""
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 20))
    
    # 1. Effect Size Plot (with standardized coefficients)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_scaled = LinearRegression().fit(X_scaled, y)
    
    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model_scaled.coef_,
        'abs_coefficient': abs(model_scaled.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    ax1 = fig.add_subplot(321)
    sns.barplot(data=coef_df.head(top_n), x='coefficient', y='feature', palette='coolwarm', ax=ax1)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'Standardized Coefficients (bit like shaply beeswarm)')
    ax1.set_xlabel('Coefficient Value (Standardized)')
    
    # 2. Predicted vs Actual Plot
    y_pred = model.predict(X)
    
    ax2 = fig.add_subplot(322)
    ax2.scatter(y_pred, y, alpha=0.5)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax2.set_title(f'Predicted vs Actual - {title}')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Actual Values')
    
    # 3. Residual Plot
    residuals = y - y_pred
    
    ax3 = fig.add_subplot(323)
    ax3.scatter(y_pred, residuals, alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title(f'Residual/Error Plot (how far the prediction was from the actual - looking for random scatter)')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    
    # 4. Variable Importance using Permutation Importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
    }).sort_values('importance', ascending=False)
    
    ax4 = fig.add_subplot(324)
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', palette='viridis', ax=ax4)
    ax4.set_title(f'Permutation Importance - {title}')
    ax4.set_xlabel('Mean Decrease in R² Score')
    
    # 5. Histogram of Residuals
    ax5 = fig.add_subplot(325)
    sns.histplot(residuals, kde=True, ax=ax5)
    ax5.set_title(f'Distribution of Residuals/Error - {title}')
    ax5.set_xlabel('Residual Value')
    
    # 6. Top Feature Partial Regression Plot (using a different approach to avoid the error)
    ax6 = fig.add_subplot(326)
    
    # Create a model with statsmodels
    X_with_const = sm.add_constant(X)
    sm_model = sm.OLS(y, X_with_const).fit()
    
    if len(X.columns) > 0:
        top_feature = coef_df.iloc[0]['feature']
        top_idx = list(X.columns).index(top_feature)
        
        # Manual partial regression plot
        # Get residuals from regressing the target variable on all other predictors
        X_others = X_with_const.drop(columns=[top_feature])
        y_resid = y - sm.OLS(y, X_others).fit().predict(X_others)
        
        # Get residuals from regressing the selected predictor on all other predictors
        x_resid = X[top_feature] - sm.OLS(X[top_feature], X_others).fit().predict(X_others)
        
        # Plot the relationship between these residuals
        ax6.scatter(x_resid, y_resid, alpha=0.5)
        
        # Add regression line
        slope, intercept = np.polyfit(x_resid, y_resid, 1)
        x_range = np.linspace(min(x_resid), max(x_resid), 100)
        ax6.plot(x_range, intercept + slope * x_range, 'r-')
        
        ax6.set_title(f'Partial Regression Plot for Top Feature: {top_feature}')
        ax6.set_xlabel(f'Residuals of {top_feature}')
        ax6.set_ylabel(f'Residuals of {title}')
    
    plt.tight_layout()
    plt.suptitle(f'Advanced Regression Analysis - {title}', fontsize=16, y=1.02)
    plt.show()
    
    return importance_df, coef_df
def visualize_feature_importance(feature_importance_df, title):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature', palette='viridis')
    plt.title(f'Top 15 Most Important Features - {title}')
    plt.xlabel('Feature Importance (Absolute Coefficient Value)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()


def multiple_regression(df, target1: str):
    # Verify that dayAndNightOf is the index
    assert df.index.name == 'dayAndNightOf', "DataFrame index must be 'dayAndNightOf'"
    
    df1 = df.copy()
    # df1 = df1.dropna()

    total_days = len(df1.index.unique())
    print(f"\nAnalyzing {target1}")
    print(f"Total days in dataset: {total_days}")

    X = df1.copy().drop(columns=[target1])
    y1 = df1[target1]
    
    def calculate_r2_contributions(X, y, model):
        predictions = model.predict(X)
        mean_y = y.mean()
        total_ss = ((y - mean_y) ** 2).sum()
        residuals = y - predictions
        r2_contributions = 1 - (residuals ** 2) / total_ss
        return r2_contributions, predictions, residuals

    model1 = LinearRegression().fit(X, y1)
    r2_contributions, predictions, residuals = calculate_r2_contributions(X, y1, model1)
    
    # Add predictions and errors to the original dataframe
    df1[f'{target1}_predicted'] = predictions
    df1[f'{target1}_error'] = residuals
    df1[f'{target1}_abs_error'] = abs(residuals)
    df1['r2_contribution'] = r2_contributions
    
    threshold = r2_contributions.mean() - 2 * r2_contributions.std()
    bad_days = df1[r2_contributions < threshold].index.unique()
    
    df_kept = df1[~df1.index.isin(bad_days)]
    df_removed = df1[df1.index.isin(bad_days)]
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model1.coef_)
    }).sort_values('importance', ascending=False)
    
    if len(bad_days) > 0:
        removed_percentage = (len(bad_days) / total_days) * 100
        print(f"\nRemoving {len(bad_days)} days ({removed_percentage:.1f}%) that negatively impact R²:")
        for day in bad_days:
            print(f"  - {day}")
        
        X_filtered = df_kept.drop(columns=[target1, 
                                          f'{target1}_predicted', f'{target1}_error', 
                                          f'{target1}_abs_error', 'r2_contribution'])
        y1_filtered = df_kept[target1]
        
        model1_filtered = LinearRegression().fit(X_filtered, y1_filtered)
        r2_filtered = model1_filtered.score(X_filtered, y1_filtered)
        r2_original = model1.score(X, y1)
        r2_improvement = r2_filtered - r2_original
        
        # Add filtered model predictions to kept dataframe
        filtered_predictions = model1_filtered.predict(X_filtered)
        df_kept[f'{target1}_filtered_predicted'] = filtered_predictions
        df_kept[f'{target1}_filtered_error'] = y1_filtered - filtered_predictions
        df_kept[f'{target1}_filtered_abs_error'] = abs(y1_filtered - filtered_predictions)
        
        print(f"\nFinal Statistics:")
        print(f"Total days: {total_days}")
        print(f"Days removed: {len(bad_days)} ({removed_percentage:.1f}%)")
        print(f"Original R²: {r2_original:.4f}")
        print(f"Filtered R²: {r2_filtered:.4f}")
        print(f"R² improvement: {r2_improvement:.4f}")
        
        # Add average error metrics
        print(f"Original mean absolute error: {df1[f'{target1}_abs_error'].mean():.4f}")
        print(f"Filtered mean absolute error: {df_kept[f'{target1}_filtered_abs_error'].mean():.4f}")
        error_improvement = df1[f'{target1}_abs_error'].mean() - df_kept[f'{target1}_filtered_abs_error'].mean()
        print(f"Error improvement: {error_improvement:.4f}")
    else:
        print(f"\nNo days found that significantly reduce R² for {target1}")
        print(f"Final Statistics:")
        print(f"Total days: {total_days}")
        print(f"Days removed: 0 (0.0%)")
        print(f"R²: {model1.score(X, y1):.4f}")
        print(f"Mean absolute error: {df1[f'{target1}_abs_error'].mean():.4f}")
    
    # Assert that dayAndNightOf is still the index in returned dataframes
    assert df_kept.index.name == 'dayAndNightOf', "Kept DataFrame should maintain 'dayAndNightOf' as index"
    assert df_removed.index.name == 'dayAndNightOf', "Removed DataFrame should maintain 'dayAndNightOf' as index"
    
    return df_kept, df_removed, feature_importance
