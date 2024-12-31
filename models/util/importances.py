import pandas as pd
import matplotlib.pyplot as plt

def get_importances(model, feature_names = None):
    if 'xgboost' in str(type(model)):
        return xgboost_importances(model)
    return catboost_importances(model, feature_names)

def xgboost_importances(model):
    feature_importances = model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': feature_importances.keys(),
        'Importance': feature_importances.values()
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df

def catboost_importances(model, feature_names):
    # Get feature importances
    feature_importances = model.get_feature_importance()

    # Create a DataFrame to display the feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df


def plot_importances(importance_df, model_name):

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(model_name + ' Feature Importances')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()
