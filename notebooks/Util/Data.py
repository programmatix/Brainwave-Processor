import pandas as pd
from IPython.display import display
def require_no_missing_values(df, columns):
    """
    Check if a dataframe has missing values in specific columns.
    
    Args:

    Returns:
        True if the dataframe has no missing values in the specified columns, False otherwise.
    """
    out = df[columns].isnull().sum().sum() == 0
    if not out:
        display(analyze_missing_values(df, columns))
        raise ValueError(f"Missing values found in columns: {columns}")
    return out


def analyze_missing_values(df, columns):
    """
    Analyze missing values for specific columns in day_data, using dayAndNightOf for dates.
    
    Args:
        day_data: DataFrame containing the data
        columns: List of column names to analyze
    """
    # Check which columns exist in the dataframe
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    
    if missing_cols:
        print("Columns not found in dataframe:")
        for col in missing_cols:
            print(f"  - {col}")
        print()
    
    # Analyze missing values for existing columns
    missing_stats = df[existing_cols].isnull().sum()
    missing_pct = (df[existing_cols].isnull().sum() / len(df)) * 100
    
    # Get first and last dates with missing values for each column
    # first_missing = {}
    # last_missing = {}
    
    # for col in existing_cols:
    #     missing_dates = day_data[day_data[col].isnull()]['dayAndNightOf']
    #     if len(missing_dates) > 0:
    #         first_missing[col] = missing_dates.min()
    #         last_missing[col] = missing_dates.max()
    #     else:
    #         first_missing[col] = None
    #         last_missing[col] = None
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Present Count': df[existing_cols].count(),
        'Missing Count': missing_stats,
        'Missing %': missing_pct.round(1),
        'Type': df[existing_cols].dtypes,
        # 'Sample': day_data[existing_cols].apply(lambda x: x.dropna().iloc[0] if not x.isnull().all() else None),
        # 'First Missing': pd.Series(first_missing),
        # 'Last Missing': pd.Series(last_missing)
    }).sort_values('Missing Count', ascending=False)
    
    # Only show columns with missing values
    summary_with_missing = summary[summary['Missing Count'] > 0]
    
    # if len(summary_with_missing) > 0:
    #     print("Columns with missing values:")
    #     print(summary_with_missing)
    # else:
    #     print("No missing values found in the specified columns!")
    
    # # Print total date range of data for context
    # print("\nTotal date range in dataset:")
    # print(f"First date: {day_data['dayAndNightOf'].min()}")
    # print(f"Last date:  {day_data['dayAndNightOf'].max()}")
    
    return summary


