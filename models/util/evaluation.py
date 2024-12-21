import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(md, model, X, y):
    # Predict the target values using the trained model
    predictions = model.predict(X)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, predictions)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y, predictions)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    print(f"{md.name} Mean Absolute Error (MAE): {mae}")
    print(f"{md.name} Mean Squared Error (MSE): {mse}")
    print(f"{md.name} Root Mean Squared Error (RMSE): {rmse}")

    return mae, mse, rmse

def evaluate_classification_model(md, X_train, y_train, X_val, y_val):
    model = md.model
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
    axes[0].set_title(f'{md.name} (training)')

    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    annotate_heatmap(axes[1], cm_val)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'{md.name} (validation)')

    plt.show()

def evaluate_regression_model(model, name: str, X_train, y_train, X_val, y_val):
    print("Evaluation for model: ", name)

    # Evaluate the model on the training set
    print(f"{name} Training Set Evaluation:")
    train_predictions = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    print(f"MAE: {train_mae}, MSE: {train_mse}, RMSE: {train_rmse}")

    plt.figure(figsize=(10, 7))
    plt.scatter(y_train, train_predictions, alpha=0.3)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} Predictions vs Actual (training)')
    plt.show()

    # Evaluate the model on the validation set
    print(f"{name} Validation Set Evaluation:")
    val_predictions = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_rmse = np.sqrt(val_mse)
    print(f"MAE: {val_mae}, MSE: {val_mse}, RMSE: {val_rmse}")

    # Plot the predictions vs actual values for the validation set
    plt.figure(figsize=(10, 7))
    plt.scatter(y_val, val_predictions, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} Predictions vs Actual (validation)')
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
        'Rows_Train': len(X_train),
        'MAE_Train': mae_train,
        'MAE_Val': mae_val,
        'MSE_Train': mse_train,
        'MSE_Val': mse_val,
        'RMSE_Train': rmse_train,
        'RMSE_Val': rmse_val
    }


def evaluate_regression_models(models_and_data):
    results = []
    for md in models_and_data:
        if not hasattr(md, 'is_classifier') or not md.is_classifier:
            if hasattr(md, 'models'):
                for ms in md.models:
                    result = evaluate_regression_model_quick(ms.model, ms.X_train, ms.y_train, ms.X_val, ms.y_val)
                    result['Model'] = md.name
                    result.update(ms.settings)
                    results.append(result)
            elif hasattr(md, 'model'):
                result = evaluate_regression_model_quick(md.model, md.X_train, md.y_train, md.X_val, md.y_val)
                result['Model'] = md.name
                results.append(result)

    results_df = pd.DataFrame(results)
    return results_df
