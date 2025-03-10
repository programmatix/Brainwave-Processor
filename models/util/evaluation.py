import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(model, name, X_train, y_train, X_val, y_val, is_classifier):
    if is_classifier:
        evaluate_classification_model(model, name, X_train, y_train, X_val, y_val)
    else:
        evaluate_regression_model(model, name, X_train, y_train, X_val, y_val)
        
def evaluate_classification_model(model, name, X_train, y_train, X_val, y_val):
    # print("Evaluation for model: ", md.name)
    # Evaluate the model on the training set
    # print(f"{md.name} Training Set Evaluation:")
    # train_mae, train_mse, train_rmse = evaluate_model(md, model, X_train, y_train)

    # Evaluate the model on the validation set
    # print(f"{md.name} Validation Set Evaluation:")
    # val_mae, val_mse, val_rmse = evaluate_model(md, model, X_val, y_val)

    # Predict the target values using the trained model
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    # Create DataFrames with actual and predicted values
    train_results_df = pd.DataFrame({'Actual': y_train, 'Predicted': train_predictions})
    val_results_df = pd.DataFrame({'Actual': y_val, 'Predicted': val_predictions})

    # Compute confusion matrices
    cm_train = confusion_matrix(train_results_df['Actual'], train_results_df['Predicted'])
    cm_val = confusion_matrix(val_results_df['Actual'], val_results_df['Predicted'])

    # Function to annotate heatmap with percentages
    def annotate_heatmap(ax, cm):
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = count / cm.sum() * 100
                ax.text(j, i, f'{count}\n({percentage:.1f}%)', ha='center', va='center', color='black')

    # Plot the confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    annotate_heatmap(axes[0], cm_train)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'{name} (training)')

    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    annotate_heatmap(axes[1], cm_val)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'{name} (validation)')

    plt.show()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_regression_model(model, name: str, X_train, y_train, X_val, y_val):
    print("Evaluation for model: ", name)

    # Evaluate the model on the training set
    print(f"{name} Training Set Evaluation:")
    train_predictions = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    print(f"MAE: {train_mae}, MSE: {train_mse}, RMSE: {train_rmse}")

    # Evaluate the model on the validation set
    print(f"{name} Validation Set Evaluation:")
    val_predictions = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_rmse = np.sqrt(val_mse)
    print(f"MAE: {val_mae}, MSE: {val_mse}, RMSE: {val_rmse}")

    # Calculate errors
    train_errors = y_train - train_predictions
    val_errors = y_val - val_predictions

    # Calculate standard deviation of errors for each integer value of actual
    def calculate_error_bars(y, errors):
        unique_values = np.arange(np.floor(y.min()), np.ceil(y.max()) + 1)
        means = []
        stds = []
        for val in unique_values:
            mask = (y >= val) & (y < val + 1)
            if np.any(mask):
                means.append(np.mean(errors[mask]))
                stds.append(np.std(errors[mask]))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        return unique_values, means, stds

    val_unique_values, val_means, val_stds = calculate_error_bars(y_val, val_errors)

    # Plot the predictions vs actual values and errors
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))

    # Calculate absolute errors for color coding
    train_abs_errors = np.abs(train_errors)
    val_abs_errors = np.abs(val_errors)
    
    # Normalize errors for coloring (0=black, 30+=white)
    train_colors = np.minimum(train_abs_errors / 30.0, 1.0)
    val_colors = np.minimum(val_abs_errors / 30.0, 1.0)
    
    # Create color maps - using a more colorful but still intuitive colormap
    # 'viridis' goes from dark blue (low error) to bright yellow (high error)
    train_cmap = plt.cm.viridis
    val_cmap = plt.cm.viridis

    # First plot with color-coded points based on error
    scatter0 = axes[0].scatter(y_train, train_predictions, c=train_colors, cmap=train_cmap, alpha=0.5)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{name} Predictions vs Actual (training)')
    fig.colorbar(scatter0, ax=axes[0], label='Absolute Error (0-30+)')

    # Second plot with color-coded points based on error
    scatter1 = axes[1].scatter(y_val, val_predictions, c=val_colors, cmap=val_cmap, alpha=0.5)
    axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'{name} Predictions vs Actual (validation)')
    fig.colorbar(scatter1, ax=axes[1], label='Absolute Error (0-30+)')

    axes[2].scatter(y_val, val_errors, alpha=0.3)
    axes[2].errorbar(val_unique_values, val_means, yerr=val_stds, fmt='o', color='r', ecolor='r', elinewidth=2, capsize=4)
    axes[2].axhline(0, color='k', linestyle='--', lw=2)
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Error')
    axes[2].set_title(f'{name} Errors (validation)')

    plt.show()
def evaluate_regression_model_quick(model, X_train, y_train, X_val, y_val):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)
    mse_train = mean_squared_error(y_train, pred_train)
    mse_val = mean_squared_error(y_val, pred_val)
    rmse_train = np.sqrt(mse_train)
    rmse_val = np.sqrt(mse_val)

    return {
        'MAE_Train': mae_train,
        'MAE_Val': mae_val,
        'RMSE_Train': rmse_train,
        'RMSE_Val': rmse_val,
        'Rows_Train': len(X_train),
    }

import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_classification_model_quick(model, X_train, y_train, X_val, y_val):
    pred_train = model.predict(X_train)

    accuracy_val = None
    precision_val = None
    recall_val = None
    f1_val = None
    tp_val = None
    fp_val = None
    tn_val = None
    fn_val = None


    if len(np.unique(y_val)) >= 2:
        pred_val = model.predict(X_val)
        accuracy_val = accuracy_score(y_val, pred_val)
        precision_val = precision_score(y_val, pred_val, average='binary')
        recall_val = recall_score(y_val, pred_val, average='binary')
        f1_val = f1_score(y_val, pred_val, average='binary')

        cm_val = confusion_matrix(y_val, pred_val)
        tn_val, fp_val, fn_val, tp_val = cm_val.ravel()

    accuracy_train = accuracy_score(y_train, pred_train)
    precision_train = precision_score(y_train, pred_train, average='binary')
    recall_train = recall_score(y_train, pred_train, average='binary')
    f1_train = f1_score(y_train, pred_train, average='binary')

    cm_train = confusion_matrix(y_train, pred_train)
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

    return {
        'AccTrain': accuracy_train,
        'PrecTrain': precision_train,
        'RecTrain': recall_train,
        'F1Train': f1_train,
        'AccVal': accuracy_val,
        'PrecVal': precision_val,
        'RecVal': recall_val,
        'F1Val': f1_val,
        'TPTrain': tp_train,
        'FPTrain': fp_train,
        'TNTrain': tn_train,
        'FNTrain': fn_train,
        'TPVal': tp_val,
        'FPVal': fp_val,
        'TNVal': tn_val,
        'FNVal': fn_val,
        'RowsTrain': len(X_train),
    }

def evaluate_regression_models(models_and_data):
    results = []
    for md_idx, md in enumerate(models_and_data):
        if not hasattr(md, 'is_classifier') or not md.is_classifier:
            if hasattr(md, 'models'):
                for ms_idx, ms in enumerate(md.models):
                    result = evaluate_regression_model_quick(ms.model, ms.X_train, ms.y_train, ms.X_val, ms.y_val)
                    result['Model'] = md.name
                    result['MdIdx'] = md_idx
                    result['MsIdx'] = ms_idx
                    result.update(ms.settings)
                    results.append(result)
            elif hasattr(md, 'model'):
                result = evaluate_regression_model_quick(md.model, md.X_train, md.y_train, md.X_val, md.y_val)
                result['Model'] = md.name
                results.append(result)


    results_df = pd.DataFrame(results)
    return results_df

def evaluate_classification_models(models_and_data):
    results = []
    for md_idx, md in enumerate(models_and_data):
        if hasattr(md, 'is_classifier') and md.is_classifier:
            if hasattr(md, 'models'):
                for ms_idx, ms in enumerate(md.models):
                    result =  {}
                    result['Model'] = md.name
                    result['MdIdx'] = md_idx
                    result['MsIdx'] = ms_idx
                    result.update(ms.settings)
                    result.update(evaluate_classification_model_quick(ms.model, ms.X_train, ms.y_train, ms.X_val, ms.y_val))
                    results.append(result)
            elif hasattr(md, 'model'):
                result = evaluate_classification_model_quick(md.model, md.X_train, md.y_train, md.X_val, md.y_val)
                result['Model'] = md.name
                results.append(result)


    results_df = pd.DataFrame(results)
    results_df = results_df.applymap(lambda x: str(x) if isinstance(x, float) else x)

    return results_df
