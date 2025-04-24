import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def qarma(df: pd.DataFrame, min_support: float = 0.1, min_confidence: float = 0.6) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    intervals = {}
    for col in numeric_cols:
        values = np.sort(df[col].dropna().unique())
        intervals[col] = []
        for i in range(len(values)):
            low = values[i]
            for j in range(len(values)-1, i-1, -1):
                high = values[j]
                support = df[col].between(low, high).mean()
                if support >= min_support:
                    intervals[col].append((low, high))
                    break
        maximal = []
        for (l, u) in intervals[col]:
            if not any(l2 <= l and u2 >= u and (l2, u2) != (l, u) for (l2, u2) in intervals[col]):
                maximal.append((l, u))
        intervals[col] = maximal
    bin_df = pd.DataFrame(index=df.index)
    for col, ivals in intervals.items():
        for (low, high) in ivals:
            bin_df[f"{col}[{low},{high}]"] = df[col].between(low, high).astype(int)
    frequent_itemsets = apriori(bin_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import time
from tqdm.auto import tqdm

class QARMA:
    """
    Quantitative Association Rule Mining Algorithm
    A simple implementation to find association rules in datasets with quantitative attributes.
    """
    
    def __init__(self, min_support=0.1, min_confidence=0.5):
        """
        Initialize QARMA with minimum support and confidence thresholds.
        
        Parameters:
        -----------
        min_support : float (0-1)
            Minimum support threshold for itemsets
        min_confidence : float (0-1)
            Minimum confidence threshold for rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = {}
        self.rules = []
        self.timing_logs = {}
        
    def _discretize_data(self, df, num_bins=3):
        """
        Discretize numerical attributes into bins.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data with numerical attributes
        num_bins : int
            Number of bins to discretize into
            
        Returns:
        --------
        pandas DataFrame
            Discretized dataframe
        """
        print("Discretizing data...")
        start_time = time.time()
        result = df.copy()
        constant_cols_dropped = []
        for column in tqdm(df.columns, desc="Discretizing columns"):
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].nunique() <= 1:
                    constant_cols_dropped.append(column)
                    result[f"{column}_single"] = 1
                    result = result.drop(column, axis=1)
                    continue
                
                # Get bin edges for equal-width binning
                min_val = df[column].min()
                max_val = df[column].max()
                
                # Add a small epsilon to max_val to ensure unique bin edges when min=max
                if min_val == max_val:
                    max_val += 0.001
                
                bins = np.linspace(min_val, max_val, num_bins + 1)
                
                # Create bin labels
                if num_bins == 3:
                    labels = [f"{column}_low", f"{column}_medium", f"{column}_high"]
                else:
                    labels = [f"{column}_bin{i+1}" for i in range(num_bins)]
                
                # Discretize and convert to one-hot encoding
                binned = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True, duplicates='drop')
                
                # Replace original column with one-hot columns
                dummies = pd.get_dummies(binned, prefix='', prefix_sep='')
                result = pd.concat([result.drop(column, axis=1), dummies], axis=1)
        
        print(f"Dropped {len(constant_cols_dropped)} constant column(s): {constant_cols_dropped}")
        self.timing_logs['discretize_data'] = time.time() - start_time
        return result

    def _handle_missing_values(self, df):
        """
        Handle missing values by treating them as a separate category.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with missing values marked
        """
        print("Handling missing values...")
        start_time = time.time()
        result = df.copy()
        missing_cols = []
        for column in tqdm(df.columns, desc="Processing missing values"):
            if df[column].isna().any():
                missing_cols.append(f"{column}_missing")
                result[f"{column}_missing"] = df[column].isna().astype(int)
                if pd.api.types.is_numeric_dtype(df[column]):
                    result[column] = df[column].fillna(df[column].mean())
                else:
                    result[column] = df[column].fillna(df[column].mode()[0])
        print(f"Created {len(missing_cols)} missing indicator column(s): {missing_cols}")
        self.timing_logs['handle_missing_values'] = time.time() - start_time
        return result
    
    def _get_transactions(self, df, include_threshold=0.5):
        """
        Convert DataFrame to transaction format.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        include_threshold : float
            For binary attributes, include only if value is above this threshold
            
        Returns:
        --------
        list of sets
            List of transactions where each transaction is a set of items
        """
        print("Converting to transaction format...")
        start_time = time.time()
        transactions = []
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating transactions"):
            transaction = set()
            for col in numeric_columns:
                val = row[col]
                if val > include_threshold:
                    transaction.add(col)
            transactions.append(transaction)
        
        self.timing_logs['get_transactions'] = time.time() - start_time
        return transactions
    
    def _generate_frequent_itemsets(self, transactions):
        """
        Generate frequent itemsets using the Apriori algorithm.
        
        Parameters:
        -----------
        transactions : list of sets
            List of transactions
            
        Returns:
        --------
        dict
            Dictionary mapping itemset size to list of (itemset, support) tuples
        """
        print("Generating frequent itemsets...")
        start_time = time.time()
        # Count single items
        item_counts = defaultdict(int)
        n_transactions = len(transactions)
        
        print("Counting individual items...")
        for transaction in tqdm(transactions, desc="Counting 1-itemsets"):
            for item in transaction:
                item_counts[frozenset([item])] += 1
        
        # Filter by minimum support
        k1_items = {k: v / n_transactions for k, v in item_counts.items() 
                   if v / n_transactions >= self.min_support}
        
        self.itemsets[1] = k1_items
        
        k = 2
        while k1_items and k <= 3:  # Limit to 3-itemsets for simplicity
            print(f"Finding {k}-itemsets...")
            # Generate candidate k-itemsets
            candidates = set()
            for i1, i2 in tqdm(list(combinations(k1_items.keys(), 2)), desc=f"Generating {k}-itemset candidates"):
                union = i1.union(i2)
                if len(union) == k:
                    candidates.add(union)
            
            # Count supports
            k_items = {}
            print(f"Counting support for {len(candidates)} candidate {k}-itemsets...")
            for candidate in tqdm(candidates, desc=f"Computing support for {k}-itemsets"):
                count = sum(1 for transaction in transactions if candidate.issubset(transaction))
                support = count / n_transactions
                if support >= self.min_support:
                    k_items[candidate] = support
            
            if k_items:
                self.itemsets[k] = k_items
                print(f"Found {len(k_items)} frequent {k}-itemsets")
            else:
                print(f"No frequent {k}-itemsets found")
                
            k1_items = k_items
            k += 1
        
        self.timing_logs['generate_itemsets'] = time.time() - start_time
        return self.itemsets
    
    def _generate_rules(self):
        """
        Generate association rules from frequent itemsets.
        
        Returns:
        --------
        list
            List of (antecedent, consequent, confidence, support, lift) tuples
        """
        print("Generating association rules...")
        start_time = time.time()
        rules = []
        
        # Start from itemsets with at least 2 items
        for k, itemsets in self.itemsets.items():
            if k < 2:
                continue
            
            print(f"Generating rules from {k}-itemsets...")    
            for itemset, support in tqdm(itemsets.items(), desc=f"Processing {k}-itemsets"):
                # Generate all possible antecedent/consequent splits
                for i in range(1, k):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        confidence = support / self.itemsets[len(antecedent)][antecedent]
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift
                            consequent_support = self.itemsets[len(consequent)][consequent]
                            lift = confidence / consequent_support
                            
                            rules.append((antecedent, consequent, confidence, support, lift))
        
        self.rules = sorted(rules, key=lambda x: x[4], reverse=True)  # Sort by lift
        self.timing_logs['generate_rules'] = time.time() - start_time
        print(f"Generated {len(self.rules)} rules")
        return self.rules
    
    def fit(self, df, discretize=True, num_bins=3):
        """
        Fit QARMA to a dataset.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        discretize : bool
            Whether to discretize numerical attributes
        num_bins : int
            Number of bins for discretization
            
        Returns:
        --------
        self
        """
        total_start_time = time.time()
        print(f"Starting QARMA with {len(df)} rows and {len(df.columns)} columns")
        
        # Preprocess data
        processed_df = self._handle_missing_values(df)
        
        if discretize:
            processed_df = self._discretize_data(processed_df, num_bins)
            
        # Convert to transactions
        transactions = self._get_transactions(processed_df)
        
        # Generate frequent itemsets
        self._generate_frequent_itemsets(transactions)
        
        # Generate rules
        self._generate_rules()
        
        self.timing_logs['total'] = time.time() - total_start_time
        
        # Print timing summary
        self._print_timing_summary()
        
        return self
    
    def _print_timing_summary(self):
        """Print a summary of timing information for each stage."""
        print("\nTiming Summary:")
        print("-" * 50)
        for stage, seconds in self.timing_logs.items():
            if stage != 'total':
                print(f"{stage.replace('_', ' ').title()}: {seconds:.2f} seconds ({seconds/self.timing_logs['total']*100:.1f}%)")
        print("-" * 50)
        print(f"Total execution time: {self.timing_logs['total']:.2f} seconds")
    
    def get_rules(self, n=None):
        """
        Get discovered association rules.
        
        Parameters:
        -----------
        n : int or None
            Number of top rules to return (sorted by lift)
            
        Returns:
        --------
        list
            List of (antecedent, consequent, confidence, support, lift) tuples
        """
        if n is None:
            return self.rules
        return self.rules[:n]
    
    def print_rules(self, n=None):
        """
        Print discovered association rules in a readable format.
        
        Parameters:
        -----------
        n : int or None
            Number of top rules to print (sorted by lift)
        """
        rules = self.get_rules(n)
        
        print(f"Found {len(rules)} rules with support >= {self.min_support} and confidence >= {self.min_confidence}")
        print("-" * 80)
        
        for i, (antecedent, consequent, confidence, support, lift) in enumerate(rules):
            antecedent_str = ", ".join(antecedent)
            consequent_str = ", ".join(consequent)
            
            print(f"Rule {i+1}: {antecedent_str} => {consequent_str}")
            print(f"  Support: {support:.3f}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Lift: {lift:.3f}")
            print("-" * 80)

# Example usage
if __name__ == "__main__":
    # Create an example dataset with some missing values
    np.random.seed(42)
    
    # Create sample data with age, income, and purchase behavior
    n_samples = 200
    
    data = {
        'Age': np.random.normal(40, 15, n_samples),
        'Income': np.random.normal(50000, 20000, n_samples),
        'ShoppingFrequency': np.random.normal(5, 2, n_samples)
    }
    
    # Add some categorical features
    data['LikesOnlineShopping'] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    data['PrefersPremiumBrands'] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    data['HasLoyaltyCard'] = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    # Create some relationships in the data
    # People with higher income tend to prefer premium brands
    data['PrefersPremiumBrands'] = np.where(
        data['Income'] > 60000,
        np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    )
    
    # Older people shop less frequently
    data['ShoppingFrequency'] = np.where(
        data['Age'] > 55,
        data['ShoppingFrequency'] * 0.7,
        data['ShoppingFrequency']
    )
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[np.random.choice(n_samples, size=15), 'Age'] = np.nan
    df.loc[np.random.choice(n_samples, size=20), 'Income'] = np.nan
    df.loc[np.random.choice(n_samples, size=10), 'PrefersPremiumBrands'] = np.nan
    
    # Add a column with all the same values to test the fix
    df['ConstantColumn'] = 0
    
    print("Sample data:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nMissing values count:")
    print(df.isna().sum())
    
    # Apply QARMA
    qarma = QARMA(min_support=0.1, min_confidence=0.6)
    qarma.fit(df, discretize=True, num_bins=3)
    
    print("\nQARMA Results:")
    qarma.print_rules(n=5)