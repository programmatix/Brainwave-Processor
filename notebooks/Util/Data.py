import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

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

def analyze_negative_values(df, columns):
    """
    Analyze negative values for specific columns in a DataFrame.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to analyze
    
    Returns:
        DataFrame with statistics about negative values
    """
    # Check which columns exist in the dataframe
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    
    if missing_cols:
        print("Columns not found in dataframe:")
        for col in missing_cols:
            print(f"  - {col}")
        print()
    
    # Filter to only numeric columns
    numeric_cols = [col for col in existing_cols if pd.api.types.is_numeric_dtype(df[col])]
    non_numeric_cols = [col for col in existing_cols if col not in numeric_cols]
    
    if non_numeric_cols:
        print("Non-numeric columns that will be skipped:")
        for col in non_numeric_cols:
            print(f"  - {col}")
        print()
    
    # Analyze negative values for existing numeric columns
    negative_stats = {col: (df[col] < 0).sum() for col in numeric_cols}
    negative_pct = {col: ((df[col] < 0).sum() / len(df)) * 100 for col in numeric_cols}
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Total Count': df[numeric_cols].count(),
        'Negative Count': pd.Series(negative_stats),
        'Negative %': pd.Series(negative_pct).round(1),
        'Type': df[numeric_cols].dtypes,
        'Min Value': df[numeric_cols].min()
    }).sort_values('Negative Count', ascending=False)
    
    return summary

def remove_missing_rows_sequentially(df, columns):
    """
    Sequentially remove rows missing each column and record removal stats.
    """
    cols = [col for col in columns if col in df.columns]
    summary_initial = analyze_missing_values(df, columns)
    present_counts = summary_initial['Present Count']
    order = present_counts.sort_values(ascending=False).index.tolist()
    records = []
    cumulative_removed = 0
    current_df = df.copy()
    for col in tqdm(order, desc="Sequential removal"):
        missing_mask = current_df[col].isnull()
        removed = missing_mask.sum()
        current_df = current_df[~missing_mask]
        cumulative_removed += removed
        records.append({'Column': col, 'Removed': removed, 'Cumulative Removed': cumulative_removed})
    return pd.DataFrame(records)

def remove_missing_rows_greedy(df, columns):
    """
    Greedily add keys in order of least row removal impact, batch zero-removal keys, and record stats with percentages.
    """
    cols = [col for col in columns if col in df.columns]
    original_len = len(df)
    current_df = df.copy()
    records = []
    cumulative_removed = 0
    remaining = set(cols)
    with tqdm(total=len(cols), desc="Greedy removal") as pbar:
        while remaining:
            removals = {col: current_df[col].isnull().sum() for col in remaining}
            zero_cols = [col for col, m in removals.items() if m == 0]
            if zero_cols:
                for zcol in zero_cols:
                    records.append({
                        'Column': zcol,
                        'Removed': 0,
                        'Removed %': 0.0,
                        'Cumulative Removed': cumulative_removed,
                        'Cumulative Removed %': round(cumulative_removed / original_len * 100, 1)
                    })
                remaining -= set(zero_cols)
                pbar.update(len(zero_cols))
                continue
            next_col = min(removals, key=removals.get)
            removed = removals[next_col]
            current_df = current_df[current_df[next_col].notnull()]
            cumulative_removed += removed
            records.append({
                'Column': next_col,
                'Removed': removed,
                'Removed %': round(removed / original_len * 100, 1),
                'Cumulative Removed': cumulative_removed,
                'Cumulative Removed %': round(cumulative_removed / original_len * 100, 1)
            })
            remaining.remove(next_col)
            pbar.update(1)
    return pd.DataFrame(records)


def group_columns_by_prefix(df: pd.DataFrame, delimiter: str = ':') -> pd.DataFrame:
    prefixes = df.columns.str.split(delimiter).str[0]
    records = []
    for prefix in pd.unique(prefixes):
        cols = df.columns[prefixes == prefix]
        group_df = df[cols]
        column_count = len(cols)
        total_cells = group_df.shape[0] * column_count
        present_count = group_df.count().sum()
        missing_count = group_df.isnull().sum().sum()
        missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        records.append({
            'prefix': prefix,
            'column_count': column_count,
            'present_count': present_count,
            'missing_count': missing_count,
            'missing_%': round(missing_pct, 1)
        })
    return pd.DataFrame.from_records(records)

def group_rows_by_prefix(df: pd.DataFrame, delimiter: str = ':') -> pd.DataFrame:
    prefixes = df.index.str.split(delimiter).str[0]
    records = []
    for prefix in pd.unique(prefixes):
        rows = df[prefixes == prefix]
        row_count = len(rows)
        present_count = rows.count().sum()
        missing_count = rows.isnull().sum().sum()
        missing_pct = (missing_count / row_count) * 100 if row_count > 0 else 0
        records.append({
            'prefix': prefix,
            'row_count': row_count,
            'present_count': present_count,
            'missing_count': missing_count,
            'missing_%': round(missing_pct, 1)
        })
    return pd.DataFrame.from_records(records)

