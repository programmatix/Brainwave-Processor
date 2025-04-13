import pandas as pd

def analyze_missing_values(day_data, columns):
    """
    Analyze missing values for specific columns in day_data, using dayAndNightOf for dates.
    
    Args:
        day_data: DataFrame containing the data
        columns: List of column names to analyze
    """
    # Check which columns exist in the dataframe
    existing_cols = [col for col in columns if col in day_data.columns]
    missing_cols = [col for col in columns if col not in day_data.columns]
    
    if missing_cols:
        print("Columns not found in dataframe:")
        for col in missing_cols:
            print(f"  - {col}")
        print()
    
    # Analyze missing values for existing columns
    missing_stats = day_data[existing_cols].isnull().sum()
    missing_pct = (day_data[existing_cols].isnull().sum() / len(day_data)) * 100
    
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
        'Present Count': day_data[existing_cols].count(),
        'Missing Count': missing_stats,
        'Missing %': missing_pct.round(1),
        'Type': day_data[existing_cols].dtypes,
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
