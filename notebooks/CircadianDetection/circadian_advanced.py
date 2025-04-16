import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import wittgenstein as wt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime, timedelta

def quantitative_associative_rule_mining(df, numeric_cols=None, categorical_cols=None, bins=5, min_support=0.1, min_confidence=0.5, min_lift=1.0):
    """
    Performs Quantitative Associative Rule Mining on circadian data.
    
    Quantitative ARM extends traditional association rule mining to handle 
    numeric attributes by discretizing them into bins before mining rules.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset containing both numeric and categorical variables
    numeric_cols : list, default=None
        List of numeric column names to include in the analysis
        If None, all numeric columns will be used
    categorical_cols : list, default=None
        List of categorical column names to include in the analysis
        If None, all object/category columns will be used
    bins : int or dict, default=5
        Number of bins for discretization or dictionary mapping column names to bin counts
    min_support : float, default=0.1
        Minimum support threshold for frequent itemsets
    min_confidence : float, default=0.5
        Minimum confidence threshold for rules
    min_lift : float, default=1.0
        Minimum lift threshold for rules
    
    Returns:
    --------
    tuple:
        - discretized_df: The discretized dataframe
        - frequent_itemsets: DataFrame of frequent itemsets
        - rules: DataFrame of discovered rules
        - rule_interpretations: List of human-readable rule interpretations
    """
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    import pandas as pd
    import numpy as np
    
    df_copy = df.copy()
    
    # Identify column types if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Processing {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    
    # Discretize numeric columns
    discretized_df = df_copy.copy()
    bin_labels = {}
    
    for col in numeric_cols:
        if col not in df_copy.columns:
            continue
            
        num_bins = bins[col] if isinstance(bins, dict) and col in bins else bins
        
        # Create readable bin labels
        col_min = df_copy[col].min()
        col_max = df_copy[col].max()
        bin_edges = np.linspace(col_min, col_max, num_bins + 1)
        
        # Create human-readable labels for the bins
        labels = [f"{col}_{i+1}" for i in range(num_bins)]
        bin_labels[col] = {label: (bin_edges[i], bin_edges[i+1]) 
                           for i, label in enumerate(labels)}
        
        # Discretize the column
        discretized_df[col] = pd.cut(df_copy[col], bins=num_bins, labels=labels)
    
    # Convert categorical columns to string to ensure compatibility
    for col in categorical_cols:
        if col in df_copy.columns:
            discretized_df[col] = discretized_df[col].astype(str)
    
    # Create transactions from discretized data
    transactions = []
    for _, row in discretized_df.iterrows():
        transaction = []
        for col in numeric_cols + categorical_cols:
            if col in discretized_df.columns and not pd.isna(row[col]):
                transaction.append(f"{col}={row[col]}")
        transactions.append(transaction)
    
    # Extract frequent itemsets and rules
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print("Mining frequent itemsets...")
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    print("Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="confidence", 
                             min_threshold=min_confidence)
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    # Sort by lift
    rules = rules.sort_values('lift', ascending=False)
    
    # Create interpretable rule descriptions
    rule_interpretations = []
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        # Make the rule more readable
        antecedent_str = " AND ".join([str(item) for item in antecedents])
        consequent_str = " AND ".join([str(item) for item in consequents])
        
        interpretation = (f"If {antecedent_str}, then {consequent_str} "
                         f"(support={rule['support']:.3f}, "
                         f"confidence={rule['confidence']:.3f}, "
                         f"lift={rule['lift']:.3f})")
        
        rule_interpretations.append(interpretation)
    
    # Print top rules
    print(f"\nFound {len(rules)} rules. Top 10 rules by lift:")
    for i, interp in enumerate(rule_interpretations[:10]):
        print(f"{i+1}. {interp}")
    
    return discretized_df, frequent_itemsets, rules, rule_interpretations

def apply_qarm_to_circadian(df_lep, target_col='LEP:datetimeSSM'):
    """
    Apply Quantitative Associative Rule Mining to circadian data.
    
    Parameters:
    -----------
    df_lep : pandas DataFrame
        DataFrame with circadian data
    target_col : str, default='LEP:datetimeSSM'
        Target column for analysis
    
    Returns:
    --------
    tuple:
        - rules: DataFrame of discovered rules
        - rule_interpretations: List of human-readable rule interpretations
    """
    # Prepare data - focus on numeric features 
    df = df_lep.copy()
    
    # Create additional time-related features
    if target_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[target_col]):
        df['LEP_hour'] = df[target_col].dt.hour + df[target_col].dt.minute/60
        df['LEP_shift'] = df['LEP_hour'].diff().fillna(0)
        
        # Create LEP earliness/lateness features
        median_lep = df['LEP_hour'].median()
        df['LEP_earliness'] = np.maximum(0, median_lep - df['LEP_hour']) 
        df['LEP_lateness'] = np.maximum(0, df['LEP_hour'] - median_lep)
    
    # Select relevant columns for QARM
    time_cols = [col for col in df.columns if ':datetime' in col or ':time' in col]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Custom bin sizes for different feature types
    bin_dict = {
        'LEP_hour': 8,          # 3-hour chunks
        'LEP_shift': 10,        # Finer granularity for shifts
        'LEP_earliness': 6,     
        'LEP_lateness': 6,
        'steps:sum': 5,         # Activity level bins
        'heartrate:mean': 5,    # Heart rate level bins
    }
    
    # Apply quantitative ARM
    _, frequent_itemsets, rules, rule_interpretations = quantitative_associative_rule_mining(
        df,
        numeric_cols=numeric_cols,
        bins=bin_dict,
        min_support=0.05,  
        min_confidence=0.6,
        min_lift=1.2
    )
    
    # Analyze specific relationships for LEP shifts
    lep_rules = rules[rules.apply(
        lambda x: any('LEP_shift' in str(item) or 'LEP_hour' in str(item) 
                     for item in list(x['antecedents']) + list(x['consequents'])), 
        axis=1
    )]
    
    print("\nRules related to LEP shifts:")
    for i, rule in lep_rules.head(10).iterrows():
        ant = " AND ".join([str(item) for item in list(rule['antecedents'])])
        cons = " AND ".join([str(item) for item in list(rule['consequents'])])
        print(f"Rule: {ant} → {cons}")
        print(f"Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}\n")
    
    return rules, rule_interpretations


def apply_ripper_to_circadian(df_lep):
    """
    Apply RIPPER algorithm to discover rules in circadian data.
    """
    # Prepare data
    # First ensure the data is sorted by date
    if 'dayAndNightOf' in df_lep.columns:
        df_lep = df_lep.sort_values('dayAndNightOf')
    
    # Create target variable - LEP shift in minutes
    df_lep['LEP_shift'] = df_lep['LEP:datetime'].diff() / 60
    
    # Drop first row with NaN shift and any other NaNs
    df_clean = df_lep.dropna(subset=['LEP_shift'])
    
    # Prepare features - select numeric features and rename for clarity
    feature_mapping = {
        'sunlightBeforeMidday': 'morning_sun_secs',
        'luminette:duration': 'luminette_secs',
        'shower:last': 'shower_time_ssm',
        'luminette:first': 'luminette_time_ssm',
        'sunlightWithin2HoursOfWake': 'early_sun_secs',
        'totalTimeAnySun': 'total_sun_secs',
        'shower:count': 'shower_count'
    }
    
    # Create simplified feature set with renamed columns
    X = pd.DataFrame()
    for old_name, new_name in feature_mapping.items():
        if old_name in df_clean.columns:
            X[new_name] = df_clean[old_name]
    
    # Create simple binary target variables
    y = df_clean['LEP_shift']
    
    # Print data overview
    print(f"Data overview: {len(X)} rows, {X.columns.size} features")
    print(f"LEP shift stats: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}, median={y.median():.1f}")
    
    # Create binary targets with clear thresholds
    # We'll try multiple thresholds to see which produce rules
    targets = {
        'advance_5min': (y < -5).astype(int),    # Phase advance >5 minutes
        'advance_10min': (y < -10).astype(int),  # Phase advance >10 minutes
        'delay_5min': (y > 5).astype(int),       # Phase delay >5 minutes
        'delay_10min': (y > 10).astype(int),     # Phase delay >10 minutes
        'stable': (abs(y) < 5).astype(int)       # Stable within ±5 minutes
    }
    
    # Print target distributions
    print("\nTarget distributions:")
    for name, target in targets.items():
        counts = target.value_counts()
        pos_count = counts.get(1, 0)
        neg_count = counts.get(0, 0)
        print(f"{name}: {pos_count} positive examples, {neg_count} negative examples")
    
    # Convert time-based features to more intuitive values
    X_processed = X.copy()
    
    if 'shower_time_ssm' in X_processed.columns:
        # Convert seconds since midnight to hours (more intuitive)
        X_processed['shower_hour'] = X_processed['shower_time_ssm'] / 3600
        X_processed.drop('shower_time_ssm', axis=1, inplace=True)
    
    if 'luminette_time_ssm' in X_processed.columns:
        X_processed['luminette_hour'] = X_processed['luminette_time_ssm'] / 3600
        X_processed.drop('luminette_time_ssm', axis=1, inplace=True)
    
    # Convert seconds to minutes for duration features
    for col in X_processed.columns:
        if col.endswith('_secs'):
            new_col = col.replace('_secs', '_mins')
            X_processed[new_col] = X_processed[col] / 60
            X_processed.drop(col, axis=1, inplace=True)
    
    # Discretize features with proper bins
    X_disc = pd.DataFrame()
    
    for col in X_processed.columns:
        print(f"Processing {col}...")
        # Skip if too few unique values
        if X_processed[col].nunique() < 3:
            X_disc[col] = X_processed[col]
            continue
            
        # Get distribution info for binning
        non_zero_vals = X_processed[col][X_processed[col] > 0]
        
        # Handle time-of-day features (hour variables)
        if 'hour' in col:
            # Time of day bins (early/morning/later)
            try:
                X_disc[f"{col}_cat"] = pd.cut(
                    X_processed[col],
                    bins=[0, 7, 10, 24],  # Early morning, Morning, Later
                    labels=['early', 'morning', 'later'],
                    duplicates='drop'
                )
                print(f"  Binned into time categories: {X_disc[f'{col}_cat'].value_counts().to_dict()}")
            except Exception as e:
                print(f"  Error binning {col} into time categories: {e}")
                # Fallback to simpler binning
                try:
                    X_disc[f"{col}_cat"] = pd.qcut(
                        X_processed[col], 
                        q=3, 
                        labels=['early', 'mid', 'late'],
                        duplicates='drop'
                    )
                    print(f"  Fallback quantile binning: {X_disc[f'{col}_cat'].value_counts().to_dict()}")
                except Exception as e2:
                    print(f"  Fallback binning also failed: {e2}")
        
        # Handle duration features that might contain zeros
        elif 'mins' in col or '_count' in col:
            # Special handling for features with many zeros
            zero_mask = X_processed[col] == 0
            zero_count = zero_mask.sum()
            
            if zero_count > 0 and zero_count < len(X_processed):
                # Create a binary indicator for zero vs non-zero
                X_disc[f"{col}_used"] = (~zero_mask).astype(str)
                print(f"  Binary usage: {X_disc[f'{col}_used'].value_counts().to_dict()}")
                
                # For non-zero values, create intensity levels
                if len(non_zero_vals) >= 3:
                    try:
                        # Try to use quantile binning for more even distribution
                        X_disc[f"{col}_amt"] = pd.qcut(
                            non_zero_vals, 
                            q=2, 
                            labels=['low', 'high'],
                            duplicates='drop'
                        )
                        # Fill zeros with "none"
                        X_disc[f"{col}_amt"] = X_disc[f"{col}_amt"].cat.add_categories(['none'])
                        X_disc.loc[zero_mask, f"{col}_amt"] = 'none'
                        print(f"  Amount categories: {X_disc[f'{col}_amt'].value_counts().to_dict()}")
                    except Exception as e:
                        print(f"  Error in quantile binning for non-zero values: {e}")
            else:
                # If almost all values are the same, just use the binary feature
                X_disc[f"{col}_used"] = (X_processed[col] > 0).astype(str)
                print(f"  Binary only: {X_disc[f'{col}_used'].value_counts().to_dict()}")
    
    # Clean up the discretized data
    X_disc = X_disc.select_dtypes(exclude=['float64', 'int64'])  # Keep only categorical features
    X_disc = X_disc.dropna(axis=1)  # Drop any columns with NaN values
    
    print(f"\nDiscretized features: {X_disc.columns.tolist()}")
    print(f"Final shape: {X_disc.shape}")
    
    # Train RIPPER models for each target
    results = {}
    all_rules = []
    
    for target_name, target_series in targets.items():
        print(f"\nTraining model for {target_name}...")
        print(f"Target distribution: {target_series.value_counts().to_dict()}")
        
        # Skip if there are too few positive examples
        if target_series.sum() < 5:
            print(f"Skipping {target_name} - not enough positive examples")
            continue
            
        # Create training data
        train_data = pd.concat([X_disc, target_series.rename('target')], axis=1)
        
        # Train RIPPER
        ripper = wt.RIPPER(k=2, prune_size=0.33)
        try:
            ripper.fit(train_data, class_feat='target', pos_class=1)
            
            # Check if we got any rules
            if len(ripper.ruleset_.rules) > 0:
                results[target_name] = ripper
                
                # Save rule details
                for rule in ripper.ruleset_.rules:
                    rule_text = str(rule).replace(' -> target=1', '')
                    all_rules.append({
                        'target': target_name,
                        'rule': rule_text,
                        'description': f"IF {rule_text} THEN {target_name}"
                    })
                
                print(f"Found {len(ripper.ruleset_.rules)} rules for {target_name}")
                print(ripper.ruleset_)
            else:
                print(f"No rules found for {target_name}")
                
        except Exception as e:
            print(f"Error training model for {target_name}: {str(e)}")
    
    # Create rules dataframe
    if all_rules:
        rules_df = pd.DataFrame(all_rules)
        
        # Create feature importance based on rule frequency
        feature_importance = {}
        for rule in rules_df['rule']:
            for feature in X_disc.columns:
                if feature in rule:
                    if feature not in feature_importance:
                        feature_importance[feature] = 0
                    feature_importance[feature] += 1
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Rule Count': list(feature_importance.values())
        }).sort_values('Rule Count', ascending=False)
        
        # Plot feature importance
        if not importance_df.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Rule Count'])
            plt.xlabel('Number of Rules Using Feature')
            plt.title('Feature Importance in Circadian Rules')
            plt.tight_layout()
            plt.show()
        
        # Create practical guidance based on the rules
        print("\n===== CIRCADIAN RHYTHM CONTROL GUIDELINES =====")
        
        # Group rules by target
        for target in rules_df['target'].unique():
            target_rules = rules_df[rules_df['target'] == target]
            
            if target.startswith('advance'):
                print(f"\nTo ADVANCE your rhythm (make LEP earlier):")
            elif target.startswith('delay'):
                print(f"\nTo DELAY your rhythm (make LEP later):")
            elif target == 'stable':
                print(f"\nTo STABILIZE your rhythm (minimize shifts):")
            
            for i, rule in enumerate(target_rules['rule']):
                print(f"{i+1}. IF {rule}")
    else:
        rules_df = pd.DataFrame()
        feature_importance = {}
        print("\nNo rules were discovered. This could mean:")
        print("1. The discretization approach didn't capture meaningful patterns")
        print("2. There are no simple rules that predict circadian shifts in this data")
        print("3. More complex interactions exist that RIPPER can't capture")
        print("\nConsider trying another approach like Random Forests or Gradient Boosting.")
    
    return {
        'models': results,
        'rules': rules_df,
        'feature_importance': feature_importance
    }


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# RuleFit implementation
class RuleFit:
    def __init__(self, max_rules=100, max_depth=3):
        self.max_depth = max_depth
        self.max_rules = max_rules
        self.rules = []
        self.rule_importances = []
        self.linear_coefs = None
        self.feature_names = None
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=self.max_depth,
            random_state=42
        )
        self.linear_model = LinearRegression()
        
    def _extract_rules_from_trees(self, X):
        """Extract decision rules from tree ensemble"""
        rules = []
        rule_ensembles = []
        
        # Get feature names if not provided
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
            
        # Extract rules from each tree in the ensemble
        for tree_idx, estimator in enumerate(self.gb_model.estimators_):
            tree = estimator[0].tree_
            
            # Function to recursively extract rules from decision tree nodes
            def extract_rules_from_node(node_id=0, conditions=[], rule_ensemble=[]):
                # If leaf node, return the rule
                if tree.children_left[node_id] == -1:  # Leaf node
                    if conditions:
                        rule_text = " AND ".join(conditions)
                        prediction = tree.value[node_id][0][0]
                        rules.append((rule_text, prediction))
                        rule_ensemble.append((conditions.copy(), prediction))
                    return
                
                # Get feature and threshold for this node
                feature = self.feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                
                # Left branch - feature <= threshold
                left_conditions = conditions.copy()
                left_conditions.append(f"{feature} <= {threshold:.3f}")
                extract_rules_from_node(
                    tree.children_left[node_id], 
                    left_conditions,
                    rule_ensemble
                )
                
                # Right branch - feature > threshold
                right_conditions = conditions.copy()
                right_conditions.append(f"{feature} > {threshold:.3f}")
                extract_rules_from_node(
                    tree.children_right[node_id], 
                    right_conditions,
                    rule_ensemble
                )
            
            # Extract rules from this tree
            current_rule_ensemble = []
            extract_rules_from_node(0, [], current_rule_ensemble)
            rule_ensembles.append(current_rule_ensemble)
            
        return rules, rule_ensembles
    
    def _create_rule_features(self, X, rule_ensembles):
        """Create binary features for each rule"""
        rule_features = np.zeros((X.shape[0], len(self.rules)))
        
        for i, (rule_text, _) in enumerate(self.rules):
            # Parse the rule text to create a boolean mask
            conditions = rule_text.split(" AND ")
            mask = np.ones(X.shape[0], dtype=bool)
            
            for condition in conditions:
                feature, op_val = condition.split(" ", 1)
                op, val = op_val.split(" ")
                val = float(val)
                
                if op == "<=":
                    mask &= (X[feature].values <= val)
                elif op == ">":
                    mask &= (X[feature].values > val)
            
            rule_features[:, i] = mask
            
        return rule_features
    
    def fit(self, X, y):
        """Fit the RuleFit model"""
        # Step 1: Train gradient boosting model to get rules
        self.gb_model.fit(X, y)
        
        # Step 2: Extract rules from trees
        all_rules, rule_ensembles = self._extract_rules_from_trees(X)
        
        # Limit to max_rules by selecting those with highest predictions
        sorted_rules = sorted(all_rules, key=lambda x: abs(x[1]), reverse=True)
        self.rules = sorted_rules[:self.max_rules]
        
        # Step 3: Create rule features
        rule_features = self._create_rule_features(X, rule_ensembles)
        
        # Step 4: Combine with linear terms for original features
        X_scaled = StandardScaler().fit_transform(X)
        X_combined = np.hstack([rule_features, X_scaled])
        
        # Step 5: Train linear model on combined features
        self.linear_model.fit(X_combined, y)
        
        # Extract and store coefficients
        rule_coefs = self.linear_model.coef_[:len(self.rules)]
        linear_coefs = self.linear_model.coef_[len(self.rules):]
        
        # Calculate rule importances (coefficient * stdev of rule feature)
        rule_stdevs = np.std(rule_features, axis=0)
        self.rule_importances = np.abs(rule_coefs * rule_stdevs)
        
        self.linear_coefs = dict(zip(X.columns, linear_coefs))
        
        return self
    
    def predict(self, X):
        """Make predictions with the RuleFit model"""
        # Create rule features
        rule_features = self._create_rule_features(X, None)
        
        # Combine with linear terms
        X_scaled = StandardScaler().fit_transform(X)
        X_combined = np.hstack([rule_features, X_scaled])
        
        # Make predictions
        return self.linear_model.predict(X_combined)
    
    def get_rules(self, top_n=None):
        """Get the most important rules"""
        if top_n is None:
            top_n = len(self.rules)
        
        # Sort rules by importance
        sorted_idx = np.argsort(self.rule_importances)[::-1]
        
        top_rules = []
        for i in sorted_idx[:top_n]:
            rule_text, prediction = self.rules[i]
            importance = self.rule_importances[i]
            top_rules.append({
                'rule': rule_text,
                'prediction': prediction,
                'importance': importance,
                'coefficient': self.linear_model.coef_[i]
            })
            
        return pd.DataFrame(top_rules)
    
    def get_linear_terms(self):
        """Get the linear term coefficients"""
        return pd.Series(self.linear_coefs).sort_values(key=abs, ascending=False)
    
    def plot_importance(self, top_n=20):
        """Plot importance of top rules and linear terms"""
        # Get top rules
        top_rules = self.get_rules(top_n)
        
        # Get linear terms
        linear_terms = self.get_linear_terms()
        linear_terms = linear_terms.iloc[:top_n] if len(linear_terms) > top_n else linear_terms
        
        # Combine and sort
        rule_names = [f"Rule {i+1}: {row['rule'][:50]}..." if len(row['rule']) > 50 else f"Rule {i+1}: {row['rule']}" 
                     for i, (_, row) in enumerate(top_rules.iterrows())]
        rule_importances = top_rules['importance'].values
        
        linear_names = [f"Linear: {name}" for name in linear_terms.index]
        linear_importances = np.abs(linear_terms.values)
        
        all_names = rule_names + linear_names
        all_importances = np.concatenate([rule_importances, linear_importances])
        
        # Sort by importance
        sort_idx = np.argsort(all_importances)[::-1][:top_n]
        sorted_names = [all_names[i] for i in sort_idx]
        sorted_importances = all_importances[sort_idx]
        
        # Plot
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        colors = ['#3498db' if 'Rule' in name else '#e74c3c' for name in sorted_names]
        plt.barh(range(len(sorted_names)), sorted_importances, color=colors)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance')
        plt.title('Rule and Linear Term Importance')
        plt.tight_layout()
        plt.show()


# Apply RuleFit to circadian data
def apply_rulefit_to_circadian(df_lep):
    # Prepare data
    # Create target: LEP shift from previous day (in minutes)
    # First ensure the data is sorted by date
    if 'dayAndNightOf' in df_lep.columns:
        df_lep = df_lep.sort_values('dayAndNightOf')
    
    # Create target variable - LEP shift in minutes
    df_lep['LEP_shift'] = df_lep['LEP:datetime'].diff() / 60
    
    # Drop first row with NaN shift and any other NaNs
    df_clean = df_lep.dropna(subset=['LEP_shift'])
    
    # Prepare features - select numeric features and rename for clarity
    feature_mapping = {
        'sunlightBeforeMidday': 'morning_sunlight',
        'luminette:duration': 'luminette_duration',
        'shower:last': 'shower_time',
        'luminette:first': 'luminette_time',
        'sunlightWithin2HoursOfWake': 'early_sunlight',
        'totalTimeAnySun': 'total_sunlight',
        'shower:count': 'shower_count'
    }
    
    # Create simplified feature set
    X = df_clean.select_dtypes(include=['float64', 'int64']).copy()
    
    # Remove target from features if present
    features_to_drop = ['LEP:datetime', 'LEP_shift']
    for col in features_to_drop:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Rename columns for better readability
    for old_name, new_name in feature_mapping.items():
        if old_name in X.columns:
            X[new_name] = X[old_name]
            X = X.drop(columns=[old_name])
    
    # Use only the most relevant features to avoid overfitting
    # Select top features by correlation with target
    y = df_clean['LEP_shift']
    feature_cors = X.apply(lambda col: abs(col.corr(y)))
    top_features = feature_cors.sort_values(ascending=False).head(10).index.tolist()
    X_selected = X[top_features]
    
    # Apply RuleFit
    print(f"Training RuleFit model with {len(top_features)} features and {len(X_selected)} samples")
    rulefit = RuleFit(max_rules=50, max_depth=3)
    rulefit.fit(X_selected, y)
    
    # Display top rules
    top_rules = rulefit.get_rules(top_n=10)
    print("\nTop 10 Rules for Predicting LEP Shift:")
    for i, (_, rule) in enumerate(top_rules.iterrows()):
        # Format the prediction as advance (negative) or delay (positive)
        effect = "advance" if rule['prediction'] < 0 else "delay"
        print(f"{i+1}. IF {rule['rule']} THEN {effect} by {abs(rule['prediction']):.1f} min")
        print(f"   Importance: {rule['importance']:.4f}, Coefficient: {rule['coefficient']:.4f}")
    
    # Display linear terms
    linear_terms = rulefit.get_linear_terms()
    print("\nLinear Term Coefficients (minutes of shift per unit change):")
    for feature, coef in linear_terms.items():
        direction = "advance" if coef < 0 else "delay"
        print(f"{feature}: {abs(coef):.4f} min {direction}")
    
    # Plot feature importances
    print("\nPlotting feature importances...")
    rulefit.plot_importance(top_n=15)
    
    # Create human-readable summary
    print("\nKey Insights for Circadian Control:")
    
    # Find the most important rule for phase advance
    advance_rules = top_rules[top_rules['prediction'] < 0]
    if not advance_rules.empty:
        best_advance = advance_rules.iloc[0]
        print(f"• Best for ADVANCING your rhythm (shifting earlier):")
        print(f"  {best_advance['rule']} → {abs(best_advance['prediction']):.1f} minutes earlier")
    
    # Find the most important rule for phase delay
    delay_rules = top_rules[top_rules['prediction'] > 0]
    if not delay_rules.empty:
        best_delay = delay_rules.iloc[0]
        print(f"• Best for DELAYING your rhythm (shifting later):")
        print(f"  {best_delay['rule']} → {best_delay['prediction']:.1f} minutes later")
    
    # Calculate threshold effects for key interventions
    morning_light_rules = [r for r in top_rules['rule'] if 'morning_sunlight' in r]
    luminette_rules = [r for r in top_rules['rule'] if 'luminette' in r]
    shower_rules = [r for r in top_rules['rule'] if 'shower' in r]
    
    if morning_light_rules:
        print(f"• Morning sunlight threshold effects detected in {len(morning_light_rules)} rules")
    
    if luminette_rules:
        print(f"• Luminette threshold effects detected in {len(luminette_rules)} rules")
    
    if shower_rules:
        print(f"• Shower timing threshold effects detected in {len(shower_rules)} rules")
    
    return rulefit, top_rules, linear_terms


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

def extract_rules_from_models(df_lep):
    # Prepare data - focus on numeric features first
    X = df_lep.select_dtypes(include=['float64', 'int64'])
    
    # Create target: LEP shift from previous day (in minutes)
    # First ensure the data is sorted by date
    if 'dayAndNightOf' in df_lep.columns:
        df_lep = df_lep.sort_values('dayAndNightOf')
    
    y = df_lep['LEP:datetime']  #.diff() / 60  # Convert seconds to minutes
    y = y.fillna(0)  # Fill first day's diff
    
    # Remove the target from features if it exists
    if 'LEP:datetime' in X.columns:
        X = X.drop(columns=['LEP:datetime'])
    
    # Create meaningful feature names by simplifying the long column names
    feature_names = {
        'sunlightBeforeMidday': 'morning_sunlight',
        'luminette:duration': 'luminette_duration',
        'shower:last': 'shower_time',
        'sunlightWithin2HoursOfWake': 'early_sunlight',
        'luminette:first': 'luminette_time'
    }
    
    # Create a copy with simplified names for the features we care most about
    X_simple = X.copy()
    for old_name, new_name in feature_names.items():
        if old_name in X_simple.columns:
            X_simple[new_name] = X_simple[old_name]
    
    # Select only the simplified columns that exist
    feature_cols = [new_name for _, new_name in feature_names.items() if new_name in X_simple.columns]
    
    if not feature_cols:
        print("None of the simplified feature names were found. Using original features.")
        feature_cols = X.columns
        X_simple = X
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_simple[feature_cols], y, test_size=0.2, random_state=42
    )
    
    # Build a random forest to get feature importances
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importances for LEP Prediction:")
    print(importance_df)
    
    # Build a simple decision tree for rule extraction
    # Use the most important features from the random forest
    top_features = importance_df.head(5)['Feature'].tolist()
    
    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    dt.fit(X_train[top_features], y_train)
    
    # Extract rules as text
    tree_rules = export_text(dt, feature_names=top_features)
    
    print("\nDecision Tree Rules for LEP Prediction:")
    print(tree_rules)
    
    return importance_df, tree_rules


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Prepare data for rule mining by discretizing continuous variables
def prepare_for_rule_mining(df):
    df_discrete = df.copy()
    
    # Discretize your target variable (LEP shift)
    # First create a shift variable (today's LEP compared to yesterday)
    df_discrete['LEP_shift'] = df_discrete['LEP:datetime'].diff()
    
    # Convert to minutes and categorize
    df_discrete['LEP_shift_mins'] = df_discrete['LEP_shift'] / 60
    df_discrete['LEP_shift_cat'] = pd.cut(
        df_discrete['LEP_shift_mins'], 
        bins=[-np.inf, -15, -5, 5, 15, np.inf],
        labels=['large_advance', 'small_advance', 'stable', 'small_delay', 'large_delay']
    )
    
    # Discretize intervention variables
    # Light exposure
    df_discrete['morning_sunlight'] = pd.cut(
        df_discrete['sunlightBeforeMidday'], 
        bins=[0, 600, 1800, np.inf],  # 0, 10min, 30min, more
        labels=['none', 'moderate', 'substantial']
    )
    
    # Luminette usage
    df_discrete['luminette_usage'] = pd.cut(
        df_discrete['luminette:duration'], 
        bins=[0, 1, 900, 1800, np.inf],  # none, 0-15min, 15-30min, more
        labels=['none', 'short', 'standard', 'extended']
    )
    
    # Shower timing relative to wake
    df_discrete['shower_timing'] = pd.cut(
        df_discrete['shower:last'], 
        bins=[-np.inf, 7200, 10800, 14400, np.inf],  # <2h, 2-3h, 3-4h, >4h after midnight
        labels=['very_early', 'early', 'mid_morning', 'late']
    )
    
    # Convert to one-hot encoding for rule mining
    categorical_cols = ['morning_sunlight', 'luminette_usage', 'shower_timing', 'LEP_shift_cat']
    one_hot = pd.get_dummies(df_discrete[categorical_cols], prefix_sep='=')
    
    return one_hot, df_discrete

# Apply rule mining
def mine_circadian_rules(df_lep):
    one_hot, df_discrete = prepare_for_rule_mining(df_lep)
    
    # We're interested in rules that predict LEP shifts
    # Separate the antecedents (interventions) from the consequents (outcomes)
    outcome_cols = [col for col in one_hot.columns if 'LEP_shift_cat' in col]
    intervention_cols = [col for col in one_hot.columns if 'LEP_shift_cat' not in col]
    
    # For each outcome, find rules that predict it
    all_rules = []
    
    for outcome in outcome_cols:
        # Add the target outcome to filter for rules predicting it
        target_df = one_hot[intervention_cols].copy()
        target_df[outcome] = one_hot[outcome]
        
        # Find frequent itemsets
        frequent_itemsets = apriori(target_df, min_support=0.05, use_colnames=True)
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        
        # Filter rules to only those predicting our outcome
        outcome_rules = rules[rules['consequents'].apply(lambda x: outcome in x)]
        
        if not outcome_rules.empty:
            all_rules.append(outcome_rules)
    
    if all_rules:
        combined_rules = pd.concat(all_rules).sort_values('lift', ascending=False)
        return combined_rules, df_discrete
    else:
        return pd.DataFrame(), df_discrete


def add_targets(base_df, full_df, target_columns):
    """
    Adds specified circadian targets to base_df from full_df and drops rows with missing targets.

    Parameters:
        base_df (pd.DataFrame): DataFrame without circadian targets
        full_df (pd.DataFrame): Original DataFrame with all circadian data
        target_columns (list): List of circadian columns to add back

    Returns:
        pd.DataFrame: base_df + targets, with rows containing missing target data dropped
    """
    df_with_targets = base_df.copy()

    print("Missing value impact per target column:")
    for col in target_columns:
        df_with_targets[col] = full_df[col]
        missing_count = df_with_targets[col].isna().sum()
        print(f"  {col}: {missing_count} rows would be dropped due to missing values")

    # Drop rows with any missing target values
    before = len(df_with_targets)
    df_with_targets.dropna(subset=target_columns, inplace=True)
    after = len(df_with_targets)
    print(f"\nTotal rows dropped due to missing target values: {before - after}")
    print(f"Remaining rows: {after} of {before}\n") 

    return df_with_targets


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats

def analyze_circadian_with_glm(df_lep, target='LEP:datetimeSSM'):
    """
    Apply Generalized Linear Models to analyze circadian rhythm data.
    Predicts the target column directly instead of shifts.
    Uses original feature names but replaces colons in formulas to avoid Patsy errors.
    """
    # Clean data - drop rows with missing target values
    df_clean = df_lep.dropna(subset=[target])
    
    # Create analysis dataframe - use original column names
    analysis_df = df_clean.copy()
    
    # Create a copy of the dataframe with sanitized column names for formulas
    formula_df = analysis_df.copy()
    
    # Create a mapping between original and sanitized column names
    col_mapping = {}
    rev_mapping = {}
    
    for col in formula_df.columns:
        # Replace colons with underscores for Patsy formulas
        new_col = col.replace(':', '_')
        col_mapping[col] = new_col
        rev_mapping[new_col] = col
        
        # Rename in the formula dataframe
        if col != new_col:
            formula_df[new_col] = formula_df[col]
            formula_df.drop(columns=[col], inplace=True)
    
    # Get sanitized target name
    target_formula = col_mapping.get(target, target)
    
    # Print data overview
    print(f"Data overview: {len(analysis_df)} rows, {analysis_df.columns.size} features")
    print(f"{target} stats: min={analysis_df[target].min():.1f}, max={analysis_df[target].max():.1f}, mean={analysis_df[target].mean():.1f}")
    
    # Create binary indicators for key interventions using sanitized feature names
    formula_df['used_luminette'] = (formula_df[col_mapping['luminette:duration']] > 0).astype(int)
    formula_df['had_shower'] = (formula_df[col_mapping['shower:count']] > 0).astype(int)
    formula_df['had_morning_shower'] = (formula_df[col_mapping['shower:count']] > 0) & (formula_df[col_mapping['shower:last']] < 12).astype(int)
    formula_df['had_evening_shower'] = (formula_df[col_mapping['shower:count']] > 0) & (formula_df[col_mapping['shower:last']] > 18).astype(int)
    formula_df['had_morning_sun'] = (formula_df[col_mapping['sunlightBeforeMidday']] > 0.1666).astype(int)  # >10 mins threshold
    
    # Additional binary indicators for new features
    if 'sunlightWithin2HoursOfSunset' in analysis_df.columns:
        formula_df['had_evening_sun'] = (formula_df[col_mapping['sunlightWithin2HoursOfSunset']] > 0).astype(int)
    
    if 'sunlightWithin2HoursOfSunrise' in analysis_df.columns:
        formula_df['had_early_morning_sun'] = (formula_df[col_mapping['sunlightWithin2HoursOfSunrise']] > 0).astype(int)
    
    if 'wentOutside' in analysis_df.columns:
        formula_df['went_outside'] = formula_df[col_mapping['wentOutside']].astype(int)
    
    # Create timing features using sanitized column names
    shower_last_col = col_mapping.get('shower:last')
    luminette_first_col = col_mapping.get('luminette:first')
    
    if shower_last_col in formula_df.columns and luminette_first_col in formula_df.columns:
        # Early morning indicator (before 8 AM)
        formula_df['early_shower'] = (formula_df[shower_last_col] < 12).astype(int)
        formula_df['early_luminette'] = (formula_df[luminette_first_col] < 12).astype(int)
        
        # Calculate time difference between interventions when both present
        mask = (formula_df['used_luminette'] == 1) & (formula_df['had_shower'] == 1)
        formula_df.loc[mask, 'lum_shower_gap'] = abs(
            formula_df.loc[mask, luminette_first_col] - 
            formula_df.loc[mask, shower_last_col]
        )
    
    # Create interaction terms
    formula_df['lum_and_shower'] = formula_df['used_luminette'] * formula_df['had_shower']
    formula_df['lum_and_sun'] = formula_df['used_luminette'] * formula_df['had_morning_sun']
    formula_df['shower_and_sun'] = formula_df['had_shower'] * formula_df['had_morning_sun']
    formula_df['all_three'] = formula_df['used_luminette'] * formula_df['had_shower'] * formula_df['had_morning_sun']
    
    # Check for evening sun interactions
    if 'had_evening_sun' in formula_df.columns:
        formula_df['evening_sun'] = formula_df['had_evening_sun']
    
    print("\nIntervention frequencies:")
    for col in ['used_luminette', 'had_shower', 'had_morning_shower', 'had_evening_shower', 'had_morning_sun', 'had_evening_sun', 
                'went_outside', 'lum_and_shower', 'lum_and_sun', 'shower_and_sun', 'all_three']:
        if col in formula_df.columns:
            print(f"{col}: {formula_df[col].sum()} days ({formula_df[col].mean()*100:.1f}%)")

    # Scale all features to mean 0, std 1, except binary indicators
    # binary_cols = ['used_luminette', 'had_shower', 'had_morning_shower', 'had_evening_shower',
    #                'early_shower', 'early_luminette',
    #             'had_morning_sun', 'had_evening_sun', 'went_outside', 'lum_and_shower', 
    #             'lum_and_sun', 'shower_and_sun', 'all_three']
    
    # Skipping scaling - it's not needed for linear regression

    # Get columns to scale (non-binary)
    # cols_to_scale = formula_df.columns[~formula_df.columns.isin(binary_cols)]
    
    # # Standardize selected columns
    # formula_df[cols_to_scale] = (formula_df[cols_to_scale] - formula_df[cols_to_scale].mean()) / formula_df[cols_to_scale].std()

    # # Assert that all cols have no na values. Do this column by column. print nice failure if not
    # for col in formula_df.columns:
    #     if formula_df[col].isna().sum() > 0:
    #         print(f"Column {col} has {formula_df[col].isna().sum()} na values")
    #         print(formula_df[col])
    #         assert False

    print("\nGLOSSARY")
    print("---------------------")
    print("Note that all the tests below, despite the fancy titles, are basically doing linear regression, with different sets of features.")
    print("coef       The actual effect of this intervention on LEP, in hours.  Positive is LEP delaying (later), negative is LEP advancing (earlier).  So -1 means it's 1 hour earlier.")
    print("P>|t|      p-value, the probability the result is significant.  < 0.05 is good (only 5% likelihood it's due to chance).")
    print("t          t-value, the number of standard deviations the coefficient is away from 0.  Higher (absolute) is more significant.  Positive is LEP delaying (later), negative is LEP advancing (earlier).")
    print("[0.025]    Lower bound of the 95% confidence interval.  With 95% confidence, the true value is at least X")
    print("[0.975]    Upper bound of the 95% confidence interval.  With 95% confidence, the true value is at most X")
    print("Intercept  Bit confused. AI says it's the value of the target when all the features are 0.")
    


    # 1. Basic Linear Model (using sanitized feature names)
    print("\n1. BASIC LINEAR MODEL")
    print("---------------------")
    
    # Map original feature names to sanitized ones for the formula
    luminette_duration_col = col_mapping.get('luminette:duration')
    shower_count_col = col_mapping.get('shower:count')
    morning_sun_col = col_mapping.get('sunlightBeforeMidday')
    
    # Fit the model with key intervention variables predicting the target directly
    basic_formula = f"{target_formula} ~ {luminette_duration_col} + {shower_count_col} + {morning_sun_col}"
    
    basic_model = smf.ols(formula=basic_formula, data=formula_df).fit()
    
    print(basic_model.summary().tables[1])  # Coefficient table only
    
    # 2. Binary Predictors Model
    print("\n2. BINARY PREDICTORS MODEL")
    print("---------------------------")
    
    # Build binary formula based on available columns
    binary_vars = ['used_luminette', 'had_shower', 'had_morning_sun']
    
    if 'had_evening_sun' in formula_df.columns:
        binary_vars.append('had_evening_sun')
    
    if 'went_outside' in formula_df.columns:
        binary_vars.append('went_outside')
    
    binary_formula = f"{target_formula} ~ " + " + ".join(binary_vars)
    binary_model = smf.ols(formula=binary_formula, data=formula_df).fit()
    
    print(binary_model.summary().tables[1])  # Coefficient table only
    
    # 3. Interaction Model
    print("\n3. INTERACTION MODEL")
    print("--------------------")
    print("Despite the fancy title, it's just adding some binary features.")
    
    # Build interaction formula based on available columns
    interaction_vars = ['used_luminette', 'had_shower', 'had_morning_sun']
    interaction_terms = []
    
    if all(var in formula_df.columns for var in ['used_luminette', 'had_shower']):
        interaction_terms.append('lum_and_shower')
    
    if all(var in formula_df.columns for var in ['used_luminette', 'had_morning_sun']):
        interaction_terms.append('lum_and_sun')
    
    if all(var in formula_df.columns for var in ['had_shower', 'had_morning_sun']):
        interaction_terms.append('shower_and_sun')
    
    interaction_formula = f"{target_formula} ~ " + " + ".join(interaction_vars + interaction_terms)
    interaction_model = smf.ols(formula=interaction_formula, data=formula_df).fit()
    
    print(interaction_model.summary().tables[1])  # Coefficient table only
    
    # Test if interactions significantly improve the model
    from statsmodels.stats.anova import anova_lm
    
    basic_vars_formula = f"{target_formula} ~ " + " + ".join(interaction_vars)
    basic_vars_model = smf.ols(formula=basic_vars_formula, data=formula_df).fit()
    
    anova_result = anova_lm(basic_vars_model, interaction_model)
    print("\nANOVA comparing models with and without interactions:")
    print(anova_result)

    print("df_resid: Degrees of freedom remaining in each model (more df = fewer parameters)")
    print("ssr: Sum of Squared Residuals (lower = better fit)")
    print("df_diff: Difference in degrees of freedom between models (3.0 means you added 3 interaction terms)")
    print("F: F-statistic testing if the improvement is significant, higher = better fit")
    print("p: p-value, < 0.05 means the interaction terms are significant")

    # 4. Time of Day Model
    print("\n4. TIME OF DAY MODEL")
    print("--------------------")
    
    # Use timing variables to predict the target
    timing_vars_orig = ['shower:first', 'shower:last', 'luminette:first', 
                     'luminette:last', 'firstEnteredOutside']
    
    # Map to sanitized column names
    timing_vars = [col_mapping.get(var) for var in timing_vars_orig if var in analysis_df.columns]
    
    # Filter to include only timing variables that are in the dataset
    available_timing_vars = [var for var in timing_vars if var in formula_df.columns]
    
    if available_timing_vars:
        # Create formula with available timing variables
        timing_formula = f"{target_formula} ~ " + " + ".join(available_timing_vars)
        
        try:
            timing_model = smf.ols(formula=timing_formula, data=formula_df).fit()
            print(timing_model.summary().tables[1])  # Coefficient table only
        except Exception as e:
            print(f"Error in time of day model: {str(e)}")
    
    # 5. Evening Light Effects Model
    print("\n5. EVENING LIGHT EFFECTS MODEL")
    print("------------------------------")
    
    # Test if evening light affects circadian phase
    evening_vars_orig = ['sunlightWithin2HoursOfSunset', 
                      'sunlightWithin1HourOfSunset']
    
    # Map to sanitized column names
    evening_vars = [col_mapping.get(var) for var in evening_vars_orig if var in analysis_df.columns]
    
    if 'had_evening_sun' in formula_df.columns:
        evening_vars.append('had_evening_sun')
    
    # Filter to include only evening variables that are in the dataset
    available_evening_vars = [var for var in evening_vars if var in formula_df.columns]
    
    if available_evening_vars:
        # Create formula with available evening variables
        evening_formula = f"{target_formula} ~ " + " + ".join(available_evening_vars)
        
        try:
            evening_model = smf.ols(formula=evening_formula, data=formula_df).fit()
            print(evening_model.summary().tables[1])  # Coefficient table only
            
            # Compare evening vs morning effects
            if all(var in formula_df.columns for var in ['had_evening_sun', 'had_morning_sun']):
                evening_days = formula_df[formula_df['had_evening_sun'] == 1][target_formula]
                morning_days = formula_df[formula_df['had_morning_sun'] == 1][target_formula]
                neither_days = formula_df[(formula_df['had_evening_sun'] == 0) & 
                                          (formula_df['had_morning_sun'] == 0)][target_formula]
                
                if len(evening_days) > 0 and len(morning_days) > 0 and len(neither_days) > 0:
                    print("\nComparison of Light Timing Effects:")
                    print(f"Evening sun only: mean={evening_days.mean():.2f}")
                    print(f"Morning sun only: mean={morning_days.mean():.2f}")
                    print(f"Neither: mean={neither_days.mean():.2f}")
        except Exception as e:
            print(f"Error in evening light model: {str(e)}")
    
    # 6. Comprehensive Model
    print("\n6. COMPREHENSIVE MODEL")
    print("----------------------")
    
    # Select the most promising features from all prior models
    # Focus on original feature names without transformations
    key_features_orig = [
        'luminette:duration',
        'shower:duration',
        'sunlightBeforeMidday',
        'sunlightWithin2HoursOfSunset',
        'shower:last',
        'luminette:first',
        'totalTimeAnySun'
    ]
    key_features_orig = df_clean.columns
    
    # Map to sanitized column names
    key_features = [col_mapping.get(var) for var in key_features_orig if var in analysis_df.columns]
    
    # Filter to include only features that are in the dataset
    available_key_features = [feat for feat in key_features if feat in formula_df.columns]
    
    if available_key_features:
        # Create formula with available key features
        comprehensive_formula = f"{target_formula} ~ " + " + ".join(available_key_features)
        
        try:
            comprehensive_model = smf.ols(formula=comprehensive_formula, data=formula_df).fit()
            print(comprehensive_model.summary().tables[1])  # Coefficient table only
        except Exception as e:
            print(f"Error in comprehensive model: {str(e)}")
    
    # 7. Lagged Effects Model (if dayAndNightOf is available)
    # if 'dayAndNightOf' in formula_df.columns:
    print("\n7. LAGGED EFFECTS MODEL")
    print("----------------------")
    
    # Sort by date
    # formula_df = formula_df.sort_values('dayAndNightOf')
    
    # Create sanitized names for key variables
    luminette_var = col_mapping.get('luminette:duration')
    shower_var = col_mapping.get('shower:duration')
    morning_sun_var = col_mapping.get('sunlightBeforeMidday')
    
    # Create lagged versions of key interventions
    if all(var in formula_df.columns for var in [luminette_var, shower_var, morning_sun_var]):
        formula_df['luminette_prev_day'] = formula_df[luminette_var].shift(1)
        formula_df['shower_prev_day'] = formula_df[shower_var].shift(1)
        formula_df['morning_sun_prev_day'] = formula_df[morning_sun_var].shift(1)
        
        # Drop rows with NaN in lagged variables
        lagged_df = formula_df.dropna(subset=['luminette_prev_day', 'shower_prev_day', 'morning_sun_prev_day'])
        
        if len(lagged_df) > 10:  # Only proceed if we have sufficient data after creating lags
            # Create formula with current and lagged variables
            lagged_formula = f"{target_formula} ~ {luminette_var} + {shower_var} + " + \
                            f"{morning_sun_var} + " + \
                            "luminette_prev_day + shower_prev_day + morning_sun_prev_day"
            
            try:
                lagged_model = smf.ols(formula=lagged_formula, data=lagged_df).fit()
                print(lagged_model.summary().tables[1])  # Coefficient table only
                
                # Print information about lagged effects
                print("\nCurrent vs. Lagged Effects (coefficients):")
                
                if luminette_var in lagged_model.params and 'luminette_prev_day' in lagged_model.params:
                    print(f"Luminette current day: {lagged_model.params[luminette_var]:.4f}")
                    print(f"Luminette previous day: {lagged_model.params['luminette_prev_day']:.4f}")
                
                if shower_var in lagged_model.params and 'shower_prev_day' in lagged_model.params:
                    print(f"Shower current day: {lagged_model.params[shower_var]:.4f}")
                    print(f"Shower previous day: {lagged_model.params['shower_prev_day']:.4f}")
                
                if morning_sun_var in lagged_model.params and 'morning_sun_prev_day' in lagged_model.params:
                    print(f"Morning sun current day: {lagged_model.params[morning_sun_var]:.4f}")
                    print(f"Morning sun previous day: {lagged_model.params['morning_sun_prev_day']:.4f}")
            except Exception as e:
                print(f"Error in lagged effects model: {str(e)}")
    
    # 8. Summary of Findings
    print("\n===== SUMMARY OF CIRCADIAN RHYTHM FACTORS =====")
    
    # Collect all models
    all_models = {
        'Basic': basic_model,
        'Binary': binary_model,
        'Interaction': interaction_model
    }
    
    # Add optional models if they exist
    try:
        all_models['Timing'] = timing_model
    except:
        pass
    
    try:
        all_models['Evening'] = evening_model
    except:
        pass
    
    try:
        all_models['Comprehensive'] = comprehensive_model
    except:
        pass
    
    try:
        all_models['Lagged'] = lagged_model
    except:
        pass
    
    # Map sanitized column names back to original names for reporting
    def get_original_name(sanitized_name):
        if sanitized_name in rev_mapping:
            return rev_mapping[sanitized_name]
        return sanitized_name  # Return as is if not found
    
    # Collect significant predictors from all models
    significant_predictors = {}
    
    for model_name, model in all_models.items():
        for var, p_value in zip(model.params.index[1:], model.pvalues[1:]):
            if p_value < 0.05:
                coef = model.params[var]
                orig_var = get_original_name(var)
                if orig_var not in significant_predictors:
                    significant_predictors[orig_var] = []
                significant_predictors[orig_var].append((model_name, coef, p_value))
    
    # Print significant predictors
    if significant_predictors:
        print("\nStatistically Significant Factors:")
        for var, results in significant_predictors.items():
            print(f"\n{var}:")
            for model_name, coef, p_value in results:
                direction = "increases" if coef > 0 else "decreases"
                print(f"  • {model_name} model: {abs(coef):.4f} ({direction}, p={p_value:.3f})")
    else:
        print("\nNo statistically significant predictors were found.")
    
    # Create practical guidelines
    print("\n===== PRACTICAL CIRCADIAN CONTROL GUIDELINES =====")
    
    # Find most influential features across models
    feature_influence = {}
    
    for model_name, model in all_models.items():
        # Calculate standardized coefficients to compare across features
        for var in model.params.index[1:]:
            # Skip if variable not in dataframe (might be intercept or derived variable)
            if var not in formula_df.columns:
                continue
                
            orig_var = get_original_name(var)
            if orig_var not in feature_influence:
                feature_influence[orig_var] = []
            
            # Store absolute value of standardized coefficient
            std_coef = abs(model.params[var]) * formula_df[var].std()
            feature_influence[orig_var].append((model_name, std_coef, model.params[var]))
    
    # Calculate average influence for each feature
    avg_influence = {}
    for var, values in feature_influence.items():
        if values:  # Only if we have values
            avg_influence[var] = (np.mean([v[1] for v in values]), np.mean([v[2] for v in values]))
    
    # Sort by average absolute influence
    sorted_features = sorted(avg_influence.items(), key=lambda x: abs(x[1][0]), reverse=True)
    
    print("\nMost influential factors (ranked):")
    for var, (abs_influence, avg_coef) in sorted_features[:10]:  # Top 10 features
        direction = "increase" if avg_coef > 0 else "decrease"
        print(f"• {var}: {abs_influence:.4f} ({direction})")
    
    # Return the analysis results
    return {
        'basic_model': basic_model,
        'binary_model': binary_model,
        'interaction_model': interaction_model,
        'all_models': all_models,
        'significant_predictors': significant_predictors,
        'feature_influence': feature_influence,
        'analysis_df': analysis_df,
        'formula_df': formula_df,
        'col_mapping': col_mapping,
        'rev_mapping': rev_mapping
    }



import featuretools as ft
import pandas as pd
import numpy as np
from featuretools.primitives import (
    TimeSince, 
    SubtractNumeric, 
    Hour, 
    Minute, 
    TimeSince, 
    TimeSincePrevious
)

def try_featuretools(df_lep):

    # Create an EntitySet properly
    es = ft.EntitySet(id="circadian")

    # Add the dataframe as an entity
    df_lep_with_index = df_lep.copy()
    # if 'dayAndNightOf' in df_lep.columns:
    #     # Use date as index if available
    #     df_lep_with_index = df_lep_with_index.reset_index(drop=True)
        
    # Add the dataframe to the EntitySet with a proper index
    es.add_dataframe(
        dataframe=df_lep_with_index,
        dataframe_name='df_lep',
        index='index',
        # make_index=True,
        # time_index='dayAndNightOf' if 'dayAndNightOf' in df_lep.columns else None
    )

    # Create custom time difference features manually
    # This is often more reliable than relying on automatic discovery
    # time_cols = [col for col in df_lep.columns if any(term in col.lower() for term in 
    #             ['ssm', 'datetime'])]
    time_cols = ['firstEnteredOutside', 'lastOutside', 'luminette:first', 'luminette:last', 'shower:first', 'shower:last', 'LEP:datetime']

    print(f"Found {len(time_cols)} time-related columns: {time_cols[:5]}...")

    # Create a custom feature to calculate differences between time columns
    # def add_time_differences(dataframe, time_columns, target_col='LEP:datetime'):
    #     """Add time differences between all time columns and the target LEP time."""
    #     result = dataframe.copy()
        
    #     for col in time_columns:
    #         if col != target_col:
    #             # Calculate hours between this event and LEP
    #             diff_name = f"{col}_to_{target_col.split(':')[-1]}_diff"
    #             result[diff_name] = result[target_col] - result[col]
        
    #     return result

    # # Apply manual feature engineering
    # df_with_diffs = add_time_differences(df_lep, time_cols)

    # Show the new features
    # new_cols = [col for col in df_with_diffs.columns if col not in df_lep.columns]
    # print(f"\nManually created {len(new_cols)} new time difference features")
    # print(f"Sample new features: {new_cols[:5]}")

    # Now run Deep Feature Synthesis with our enhanced dataset and specific primitives
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="df_lep",
        # None of these work on the basic tabular data I'm using (no relationships, each row is a unique index value)
        agg_primitives=["mean", "std", "min", "max", "count", "percent_true"],
        # trans_primitives=["time_since_previous", "subtract", "divide", "multiply"],
        trans_primitives=["diff"],
        ignore_columns={"df_lep": ["dayAndNightOf"]},  # Ignore date column to avoid errors
        # trans_primitives=[SubtractNumeric],  # Explicitly include the Subtract primitive
        cutoff_time=None,  # No time cutoff
        max_depth=2  # Increase depth for more complex features
    )

    # Combine manual features with automatically generated ones
    # combined_features = pd.concat([
    #     feature_matrix,
    #     df_with_diffs[new_cols]
    # ], axis=1)

    print(f"\nFinal feature matrix has {len(list(feature_defs))} features")
    return feature_matrix, feature_defs

    # Calculate feature correlations with target
    # if 'LEP:datetime' in df_lep.columns:
    #     target_col = 'LEP:datetime'
    #     correlations = []
        
    #     for col in combined_features.columns:
    #         if col != target_col:
    #             corr = combined_features[col].corr(df_lep[target_col])
    #             if not np.isnan(corr):
    #                 correlations.append((col, corr))
        
    #     # Sort by absolute correlation
    #     sorted_correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
        
    #     print("\nTop 10 features correlated with LEP time:")
    #     for feature, corr in sorted_correlations[:10]:
    #         direction = "increases" if corr > 0 else "decreases"
    #         print(f"• {feature}: {abs(corr):.4f} ({direction} LEP time)")

    # combined_features.head(5)    

def create_event_stream(df_lep, time_cols=None, debug=False):
    """
    Convert daily time data into a stream of events.
    
    Parameters:
        df_lep (pd.DataFrame): Dataframe with one row per day
        time_cols (list): List of time columns to convert to events
        debug (bool): Whether to print detailed error information
        
    Returns:
        list: List of (timestamp, event_name) tuples
    """
    if time_cols is None:
        time_cols = ['firstEnteredOutside', 'lastOutside', 'luminette:first', 
                     'luminette:last', 'shower:first', 'shower:last', 'LEP:datetime']
    
    # Create event stream from time columns
    event_stream = []
    error_report = {}
    
    # Track the found columns
    found_columns = []
    
    # Process each day
    for idx, row in df_lep.iterrows():
        day_events = []
        day_date = None

        # Get the date from index (dayAndNightOf is the index)
        try:
            if isinstance(idx, pd.Timestamp) or isinstance(idx, datetime):
                day_date = idx
            # If the index is a string that can be parsed as a date
            elif isinstance(idx, str):
                day_date = pd.to_datetime(idx)
            else:
                # If there's a dayAndNightOf column, try that
                day_date = pd.to_datetime(row.index)
        except Exception as e:
            if debug:
                print(f"Warning: Could not parse date from index for row {idx}, using dummy date instead. Error: {str(e)}")
            # If all conversions fail, create a dummy date
            day_date = datetime(2000, 1, 1) + timedelta(days=len(event_stream))
        
        # Process each time column
        for col in time_cols:
            if col in row and not pd.isna(row[col]):
                # Get the time value (hours past midnight)
                time_val = row[col]
                
                # Track found columns
                if col not in found_columns:
                    found_columns.append(col)
                
                # Convert to hours and minutes with validation
                try:
                    # Extract hours and minutes (handle values > 24 hours)
                    total_hours = time_val


                    hours = int(total_hours) 
                    if hours == 0:
                        continue
                    minutes = int((total_hours - int(total_hours)) * 60)
                    
                    # Adjust date and hour if hour value is >= 24
                    extra_days = 0
                    if hours >= 24:
                        extra_days = hours // 24
                        hours = hours % 24
                        
                        if debug:
                            print(f"Note: Value {time_val} (row {idx}, column {col}) interpreted as {hours}:{minutes} on day +{extra_days}")
                    
                    # Validate hours and minutes are in range after adjustment
                    if hours < 0 or hours > 23:
                        error_key = f"{col}_hour_out_of_range_after_adjustment"
                        if error_key not in error_report:
                            error_report[error_key] = []
                        error_report[error_key].append({
                            'row_index': idx,
                            'value': time_val,
                            'hours': hours,
                            'minutes': minutes,
                            'extra_days': extra_days
                        })
                        # Skip this invalid event
                        continue
                        
                    if minutes < 0 or minutes > 59:
                        # Just clamp minutes to valid range
                        minutes = min(max(0, minutes), 59)
                    
                    # Apply the date adjustment if needed
                    adjusted_date = day_date
                    if extra_days > 0:
                        adjusted_date = day_date + timedelta(days=extra_days)
                    
                    # Create timestamp
                    event_time = adjusted_date.replace(hour=hours, minute=minutes)
                    
                    # Get event name (remove :datetime or :first, etc.)
                    event_name = col.split(':')[0] if ':' in col else col
                    
                    # Add to day's events
                    day_events.append((event_time, event_name))
                except Exception as e:
                    error_key = f"{col}_parse_error"
                    if error_key not in error_report:
                        error_report[error_key] = []
                    error_report[error_key].append({
                        'row_index': idx,
                        'value': time_val,
                        'error': str(e)
                    })
        
        # Sort the day's events by time
        day_events.sort(key=lambda x: x[0])
        
        # Add to the event stream
        event_stream.extend(day_events)
    
    # Print error report
    if error_report:
        print("\n===== DATA VALIDATION ERRORS =====")
        for error_type, errors in error_report.items():
            print(f"\n{error_type} ({len(errors)} occurrences):")
            
            # Show first 5 errors
            for i, error in enumerate(errors[:5]):
                if 'hours' in error:
                    extra_days_info = f", shifted {error.get('extra_days', 0)} days forward" if 'extra_days' in error and error['extra_days'] > 0 else ""
                    print(f"  {i+1}. Row {error['row_index']}: value {error['value']} converts to {error['hours']}:{error['minutes']}{extra_days_info}")
                else:
                    print(f"  {i+1}. Row {error['row_index']}: value {error['value']} - {error['error']}")
            
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors of this type")
    
    # Report missing columns
    missing_columns = [col for col in time_cols if col not in found_columns]
    if missing_columns and debug:
        print("\nWARNING: The following requested columns were not found in the dataframe:")
        for col in missing_columns:
            print(f"  - {col}")
    
    print(f"\nProcessed {len(df_lep)} days into {len(event_stream)} events")
    
    return event_stream

def create_time_windows(event_stream, window_size=30, use_time_buckets=True):
    """
    Create sliding time windows from event stream and extract temporal patterns.
    
    Parameters:
        event_stream (list): List of (timestamp, event_name) tuples
        window_size (int): Size of time windows in minutes
        use_time_buckets (bool): Whether to bucket time differences into ranges
        
    Returns:
        list: List of transactions (sets of events in each window)
    """
    if not event_stream:
        return []
    
    # Sort events by time
    event_stream.sort(key=lambda x: x[0])
    
    # Get start and end times
    start_time = event_stream[0][0]
    end_time = event_stream[-1][0]
    
    # Create sliding windows
    windows = []
    current_time = start_time
    window_delta = timedelta(minutes=window_size)
    overlap_delta = timedelta(minutes=window_size // 2)  # 50% overlap
    
    # Define time buckets if needed
    def get_time_bucket(minutes):
        if minutes < 5:
            return "immediate"
        elif minutes < 15:
            return "under_15min"
        elif minutes < 30:
            return "15_30min"
        elif minutes < 60:
            return "30_60min"
        elif minutes < 120:
            return "1_2hours"
        else:
            return "over_2hours"
    
    while current_time < end_time:
        window_end = current_time + window_delta
        
        # Get events in this window
        window_events = []
        for event_time, event_name in event_stream:
            if current_time <= event_time < window_end:
                window_events.append(event_name)
                
                # Add sequential patterns (for events within 120 minutes of each other)
                for prev_time, prev_name in event_stream:
                    if prev_time < event_time and (event_time - prev_time).total_seconds() <= 7200:  # 2 hours
                        time_diff = int((event_time - prev_time).total_seconds() / 60)
                        
                        if use_time_buckets:
                            # Use bucketed time differences
                            bucket = get_time_bucket(time_diff)
                            window_events.append(f"{prev_name}_to_{event_name}_{bucket}")
                        else:
                            # Use exact minutes
                            window_events.append(f"{prev_name}_to_{event_name}_{time_diff}min")
        
        # Add non-empty windows
        if window_events:
            windows.append(window_events)
        
        # Slide the window forward with overlap
        current_time += overlap_delta
    
    return windows

def temporal_association_mining(df_lep, time_cols=None, window_size=30, min_support=0.1, min_confidence=0.7, use_time_buckets=True):
    """
    Convert daily time data into event streams and perform association rule mining with temporal components.
    
    Parameters:
        df_lep (pd.DataFrame): Dataframe with one row per day
        time_cols (list): List of time columns to convert to events
        window_size (int): Size of time windows in minutes
        min_support (float): Minimum support for apriori algorithm
        min_confidence (float): Minimum confidence for association rules
        use_time_buckets (bool): Whether to bucket time differences into ranges
        
    Returns:
        tuple: (event_stream, transactions, frequent_itemsets, rules)
    """
    # First create the event stream
    event_stream = create_event_stream(df_lep, time_cols)
    
    # Create time windows
    print("Creating time windows...")
    windows = create_time_windows(event_stream, window_size, use_time_buckets)
    
    # Apply association rule mining
    if windows:
        te = TransactionEncoder()
        print("Transforming windows...")
        te_ary = te.fit_transform(windows)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        print("Finding frequent itemsets...")
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        
        # Generate rules
        print("Generating rules...")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return event_stream, windows, frequent_itemsets, rules
    else:
        return event_stream, [], pd.DataFrame(), pd.DataFrame()

def interpret_rule(antecedents, consequents, confidence, lift):
    """Create a human-readable interpretation of a rule"""
    # Convert frozensets to lists
    ant_list = list(antecedents)
    cons_list = list(consequents)
    
    # Function to make time bucket names more readable
    def readable_time_bucket(item):
        if isinstance(item, str) and "_to_" in item:
            parts = item.split("_")
            if parts[-1] in ["immediate", "under_15min", "15_30min", "30_60min", "1_2hours", "over_2hours"]:
                # Get the event names
                events = item[:item.rfind("_")]
                time_bucket = parts[-1]
                
                # Make it more readable
                if time_bucket == "immediate":
                    return events.replace("_to_", " immediately followed by ")
                elif time_bucket == "under_15min":
                    return events.replace("_to_", " followed within 15 minutes by ")
                elif time_bucket == "15_30min":
                    return events.replace("_to_", " followed 15-30 minutes later by ")
                elif time_bucket == "30_60min":
                    return events.replace("_to_", " followed 30-60 minutes later by ")
                elif time_bucket == "1_2hours":
                    return events.replace("_to_", " followed 1-2 hours later by ")
                elif time_bucket == "over_2hours":
                    return events.replace("_to_", " followed over 2 hours later by ")
        return item
    
    # Make items more readable
    readable_ant = [readable_time_bucket(item) for item in ant_list]
    readable_cons = [readable_time_bucket(item) for item in cons_list]
    
    # Check if this is a sequential pattern
    sequential_patterns = [item for item in ant_list if '_to_' in item]
    
    if sequential_patterns:
        # This is a temporal sequence rule
        return f"When {', '.join(str(item) for item in readable_ant)}, then {', '.join(str(item) for item in readable_cons)} occurs with {confidence:.1%} confidence"
    else:
        # Regular association rule
        return f"If {', '.join(str(item) for item in readable_ant)}, then {', '.join(str(item) for item in readable_cons)} with {confidence:.1%} confidence"

def visualize_temporal_patterns(event_stream, rules=None):
    """
    Visualize temporal patterns and discovered rules.
    
    Parameters:
        event_stream (list): List of (timestamp, event_name) tuples
        rules (pd.DataFrame): Association rules dataframe
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    if not event_stream:
        print("No events to visualize")
        return
    
    # Extract unique event types
    event_types = list(set(event[1] for event in event_stream))
    
    # Create a color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
    color_map = dict(zip(event_types, colors))
    
    # Sort events by time
    event_stream.sort(key=lambda x: x[0])
    
    # Get unique days
    days = list(set(event[0].date() for event in event_stream))
    days.sort()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot events on timeline
    for day_idx, day in enumerate(days):
        day_events = [event for event in event_stream if event[0].date() == day]
        
        for event_time, event_name in day_events:
            # Plot point
            time_in_day = event_time.hour + event_time.minute / 60
            ax.scatter(time_in_day, day_idx, color=color_map[event_name], s=100, label=event_name)
            
            # Add label
            ax.text(time_in_day, day_idx+0.1, event_name, fontsize=8, ha='center')
    
    # Add legend (without duplicates)
    handles, labels = [], []
    for event_type in event_types:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_map[event_type], markersize=10))
        labels.append(event_type)
    
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Set labels and title
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels([day.strftime('%Y-%m-%d') for day in days])
    ax.set_xlabel('Time of Day (hours)')
    ax.set_ylabel('Date')
    ax.set_xlim(0, 24)
    ax.set_title('Temporal Event Patterns')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # If rules are provided, annotate the most significant ones
    if rules is not None and not rules.empty:
        top_rule = rules.sort_values('lift', ascending=False).iloc[0]
        ax.text(0.5, -0.1, f"Top Rule: {top_rule['interpretation']}", 
                ha='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def analyze_temporal_rules(rules, top_n=20):
    """
    Analyze and interpret temporal association rules.
    
    Parameters:
        rules (pd.DataFrame): Association rules dataframe
        top_n (int): Number of top rules to analyze
        
    Returns:
        pd.DataFrame: Annotated rules with interpretations
    """
    if rules.empty:
        return pd.DataFrame()
    
    # Sort rules by lift
    sorted_rules = rules.sort_values('lift', ascending=False).head(top_n)
    
    # Add interpretation column
    sorted_rules['interpretation'] = sorted_rules.apply(
        lambda row: interpret_rule(row['antecedents'], row['consequents'], row['confidence'], row['lift']),
        axis=1
    )
    
    # Print top rules
    print(f"\nTop {min(top_n, len(sorted_rules))} Temporal Association Rules:")
    for i, (_, rule) in enumerate(sorted_rules.iterrows()):
        print(f"{i+1}. {rule['interpretation']}")
        print(f"   Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
    
    return sorted_rules

def winepi_algorithm(event_stream, window_size_mins=60, min_support=0.01, verbose=False):
    """
    Implementation of the WINEPI algorithm for discovering temporal patterns.
    
    WINEPI is a window-based method that finds frequent episodes (ordered sequences 
    of events) in temporal data.
    
    Parameters:
        event_stream (list): List of (timestamp, event_name) tuples
        window_size (int): Size of the sliding window in minutes
        min_support (float): Minimum frequency threshold (0-1)
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (sorted_episodes, windows)
            - sorted_episodes: Dictionary of frequent episodes with their support
            - windows: List of event sequences found in time windows
    """
    if not event_stream:
        return {}, []
    
    if verbose:
        print(f"Running WINEPI with window size of {window_size_mins} minutes")
        print(f"Minimum support threshold: {min_support}")
        print(f"Processing {len(event_stream)} events\n")
    
    # Sort events by time
    event_stream.sort(key=lambda x: x[0])
    
    # Create sliding windows
    windows = []
    start_time = event_stream[0][0]
    end_time = event_stream[-1][0]
    window_delta = timedelta(minutes=window_size_mins)
    
    # Generate sliding windows with fixed step size
    step_size = window_size_mins / 4  # 75% overlap
    step_delta = timedelta(minutes=step_size)
    
    current_time = start_time
    total_windows = 0
    
    # Store window event times for later analysis
    window_event_times = []
    
    while current_time < end_time:
        window_end = current_time + window_delta
        
        # Extract events in this window
        window_events = []
        window_times = {}  # To store timestamps for events in this window
        
        for event_time, event_name in event_stream:
            if current_time <= event_time < window_end:
                window_events.append((event_time, event_name))
                
                # Store timestamp for each event
                if event_name not in window_times:
                    window_times[event_name] = []
                window_times[event_name].append(event_time)
        
        # Sort events within window by time
        window_events.sort(key=lambda x: x[0])
        
        # Extract event names only (in order)
        if window_events:
            event_sequence = [event[1] for event in window_events]
            windows.append(event_sequence)
            window_event_times.append(window_times)
            total_windows += 1
        
        # Move window forward
        current_time += step_delta
    
    if verbose:
        print(f"Created {total_windows} sliding windows")
    
    # Count episode occurrences
    episode_counts = {}
    
    # Store time gaps for each episode type
    episode_time_gaps = {}
    
    # Generate episodes of different lengths (1, 2, and 3)
    for window_idx, window in enumerate(windows):
        # Single events
        # for i in range(len(window)):
        #     episode = (window[i],)
        #     episode_counts[episode] = episode_counts.get(episode, 0) + 1
        
        # Pairs of events (order matters)
        for i in range(len(window)-1):
            for j in range(i+1, min(i+4, len(window))):  # Limit distance between events
                first_event = window[i]
                second_event = window[j]
                episode = (first_event, second_event)
                episode_counts[episode] = episode_counts.get(episode, 0) + 1
                
                # Calculate time gaps
                if episode not in episode_time_gaps:
                    episode_time_gaps[episode] = []
                
                # Get all possible time differences between these events in this window
                window_times = window_event_times[window_idx]
                if first_event in window_times and second_event in window_times:
                    for t1 in window_times[first_event]:
                        for t2 in window_times[second_event]:
                            if t1 < t2:  # Ensure correct order
                                time_diff_minutes = (t2 - t1).total_seconds() / 60
                                if time_diff_minutes <= window_size_mins:  # Only count if within window
                                    episode_time_gaps[episode].append(time_diff_minutes)
        
        # Triplets of events (order matters)
        for i in range(len(window)-2):
            for j in range(i+1, min(i+4, len(window)-1)):
                for k in range(j+1, min(j+4, len(window))):
                    first_event = window[i]
                    second_event = window[j]
                    third_event = window[k]
                    episode = (first_event, second_event, third_event)
                    episode_counts[episode] = episode_counts.get(episode, 0) + 1
                    
                    # Calculate time gaps
                    if episode not in episode_time_gaps:
                        episode_time_gaps[episode] = []
                    
                    # Get all possible time differences between these events in this window
                    window_times = window_event_times[window_idx]
                    if all(e in window_times for e in episode):
                        for t1 in window_times[first_event]:
                            for t2 in window_times[second_event]:
                                for t3 in window_times[third_event]:
                                    if t1 < t2 < t3:  # Ensure correct order
                                        time_diff_1_2 = (t2 - t1).total_seconds() / 60
                                        time_diff_2_3 = (t3 - t2).total_seconds() / 60
                                        if time_diff_1_2 + time_diff_2_3 <= window_size_mins:
                                            episode_time_gaps[episode].append((time_diff_1_2, time_diff_2_3))
    
    # Calculate support (frequency)
    episode_support = {}
    
    # Calculate statistics for each episode
    episode_stats = {}
    
    for episode, count in episode_counts.items():
        support = count / total_windows if total_windows > 0 else 0
        if support >= min_support:
            episode_support[episode] = support
            
            # Add statistics if available
            if len(episode) > 1 and episode in episode_time_gaps and episode_time_gaps[episode]:
                if len(episode) == 2:
                    time_gaps = episode_time_gaps[episode]
                    if time_gaps:
                        avg_gap = sum(time_gaps) / len(time_gaps)
                        min_gap = min(time_gaps)
                        max_gap = max(time_gaps)
                        std_gap = (sum((g - avg_gap) ** 2 for g in time_gaps) / len(time_gaps)) ** 0.5 if len(time_gaps) > 1 else 0
                        
                        episode_stats[episode] = {
                            'avg_gap_minutes': avg_gap,
                            'min_gap_minutes': min_gap,
                            'max_gap_minutes': max_gap,
                            'std_gap_minutes': std_gap,
                            'window_size': window_size_mins,
                            'count': count,
                            'support': support
                        }
    
    # Sort by support
    sorted_episodes = dict(sorted(episode_support.items(), key=lambda x: x[1], reverse=True))
    
    if verbose:
        print(f"\nFound {len(sorted_episodes)} frequent episodes with support >= {min_support}")
        print("\nTop episodes by support:")
        
        for i, (episode, support) in enumerate(list(sorted_episodes.items())[:20]):
            episode_str = " → ".join(episode)
            support_pct = support * 100
            
            # Display time gap information if available
            gap_info = ""
            if episode in episode_stats:
                stats = episode_stats[episode]
                gap_info = f", avg gap: {stats['avg_gap_minutes']:.1f}min (range: {stats['min_gap_minutes']:.1f}-{stats['max_gap_minutes']:.1f}min)"
            
            print(f"{i+1}. {episode_str}: {support_pct:.1f}% support{gap_info}")
    
    return sorted_episodes, windows, episode_stats

def explain_winepi_results(sorted_episodes, window_size, episode_stats=None, top_n=10):
    """
    Explain WINEPI results in detail with a focus on interpretation.
    
    Parameters:
        sorted_episodes (dict): Dictionary of episodes with their support values
        window_size (int): Size of the sliding window in minutes used in WINEPI
        episode_stats (dict): Dictionary of episode statistics (optional)
        top_n (int): Number of top episodes to explain
        
    Returns:
        None (prints explanation)
    """
    print("\n=== WINEPI RESULTS EXPLANATION ===\n")
    print(f"Window size used: {window_size} minutes")
    print("Support value meaning: The fraction of time windows containing this episode")
    print("Example: A support of 0.25 means this sequence appeared in 25% of all time windows\n")
    
    print("Top episodes by frequency:")
    
    for i, (episode, support) in enumerate(list(sorted_episodes.items())[:top_n]):
        if i >= top_n:
            break
            
        episode_str = " → ".join(episode)
        support_pct = support * 100
        
        print(f"{i+1}. {episode_str}: {support_pct:.1f}% support")
        
        # Interpret the results
        if len(episode) == 1:
            print(f"   Interpretation: The event '{episode[0]}' occurs in {support_pct:.1f}% of all time windows.")
        elif len(episode) >= 2:
            print(f"   Interpretation: The sequence '{episode_str}' occurs in {support_pct:.1f}% of all time windows.")
            
            # Add time gap information if available
            if episode_stats and episode in episode_stats:
                stats = episode_stats[episode]
                print(f"   Time relationship: On average, {episode[1]} occurs {stats['avg_gap_minutes']:.1f} minutes after {episode[0]}")
                print(f"   Gap range: {stats['min_gap_minutes']:.1f}-{stats['max_gap_minutes']:.1f} minutes")
                
                consistency = 1 - (stats['std_gap_minutes'] / stats['avg_gap_minutes']) if stats['avg_gap_minutes'] > 0 else 0
                consistency_desc = "very consistent" if consistency > 0.8 else "somewhat consistent" if consistency > 0.5 else "variable"
                print(f"   Consistency: {consistency_desc} ({consistency:.2f})")
            else:
                print(f"   Note: These events occur in the same {window_size}-minute window, but the exact time gap varies.")
        
        print()
    
    print("\nUnderstanding the numbers:")
    print(f"1. WINEPI uses overlapping time windows of {window_size} minutes.")
    print("2. The support value shows how frequently a pattern occurs across all windows.")
    print("3. Higher support = more frequent/reliable pattern.")
    print("4. Events in a sequence occur in order, but may have variable time gaps.")
    print("5. A → B doesn't necessarily imply causation, just temporal sequence.")

def minepi_algorithm(event_stream, max_time_gap=60, min_support=2, verbose=False):
    """
    Implementation of the MINEPI algorithm for discovering minimal occurrences of episodes.
    
    MINEPI focuses on minimal occurrences of episodes where an episode occurs 
    within a time window without being part of a larger occurrence.
    
    Parameters:
        event_stream (list): List of (timestamp, event_name) tuples
        max_time_gap (int): Maximum time between events in an episode (minutes)
        min_support (int): Minimum number of occurrences required
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (sorted_episodes, episode_details)
            - sorted_episodes: Dictionary of minimal episodes with their occurrence counts
            - episode_details: Dictionary with detailed information about each episode
    """
    if not event_stream:
        return {}, {}
    
    if verbose:
        print(f"Running MINEPI with maximum time gap of {max_time_gap} minutes")
        print(f"Minimum support threshold: {min_support} occurrences")
        print(f"Processing {len(event_stream)} events\n")
    
    # Sort events by time
    event_stream.sort(key=lambda x: x[0])
    
    # Create a more efficient data structure for lookup
    # Group events by type
    events_by_type = {}
    for event_time, event_name in event_stream:
        if event_name not in events_by_type:
            events_by_type[event_name] = []
        events_by_type[event_name].append(event_time)
    
    if verbose:
        print(f"Found {len(events_by_type)} unique event types")
        for event_type, occurrences in events_by_type.items():
            print(f"  - {event_type}: {len(occurrences)} occurrences")
        print()
    
    # Find minimal occurrences of episodes
    episode_occurrences = {}
    
    # Single event episodes
    for event_type, timestamps in events_by_type.items():
        episode = (event_type,)
        episode_occurrences[episode] = [[(timestamp, timestamp)] for timestamp in timestamps]
    
    # Filter episodes by support
    frequent_episodes = {}
    for episode, occurrences in episode_occurrences.items():
        if len(occurrences) >= min_support:
            frequent_episodes[episode] = occurrences
    
    if verbose:
        print(f"Found {len(frequent_episodes)} frequent 1-event episodes")
    
    # Candidate generation for 2-event episodes
    candidates_2 = []
    for event1 in events_by_type.keys():
        for event2 in events_by_type.keys():
            if event1 != event2:
                candidates_2.append((event1, event2))
    
    if verbose:
        print(f"Generated {len(candidates_2)} candidate 2-event episodes")
    
    # Store detailed information about each episode
    episode_details = {}
    
    # Find minimal occurrences of 2-event episodes
    for candidate in candidates_2:
        event1, event2 = candidate
        occurrences = []
        time_gaps = []
        
        for t1 in events_by_type[event1]:
            for t2 in events_by_type[event2]:
                time_diff = (t2 - t1).total_seconds() / 60  # Convert to minutes
                if 0 < time_diff <= max_time_gap:
                    # This is a minimal occurrence
                    occurrences.append([(t1, t1), (t2, t2)])
                    time_gaps.append(time_diff)
        
        if len(occurrences) >= min_support:
            episode_occurrences[candidate] = occurrences
            frequent_episodes[candidate] = occurrences
            
            # Calculate statistics for time gaps
            if time_gaps:
                avg_gap = sum(time_gaps) / len(time_gaps)
                min_gap = min(time_gaps)
                max_gap = max(time_gaps)
                std_gap = (sum((g - avg_gap) ** 2 for g in time_gaps) / len(time_gaps)) ** 0.5 if len(time_gaps) > 1 else 0
                
                # Store detailed information
                episode_details[candidate] = {
                    'avg_gap_minutes': avg_gap,
                    'min_gap_minutes': min_gap,
                    'max_gap_minutes': max_gap,
                    'std_gap_minutes': std_gap,
                    'count': len(occurrences),
                    'max_time_gap': max_time_gap,
                    'time_gaps': time_gaps,
                    'consecutive_days': count_consecutive_days(occurrences)
                }
    
    if verbose:
        print(f"Found {sum(1 for ep in frequent_episodes if len(ep) == 2)} frequent 2-event episodes")
    
    # Candidate generation for 3-event episodes
    candidates_3 = []
    for e1, e2 in candidates_2:
        for e3 in events_by_type.keys():
            if e3 != e1 and e3 != e2 and (e2, e3) in frequent_episodes.keys():
                candidates_3.append((e1, e2, e3))
    
    if verbose:
        print(f"Generated {len(candidates_3)} candidate 3-event episodes")
    
    # Find minimal occurrences of 3-event episodes
    for candidate in candidates_3:
        event1, event2, event3 = candidate
        occurrences = []
        time_gaps = []
        
        for occ_12 in episode_occurrences.get((event1, event2), []):
            _, t2 = occ_12[1]  # End time of the second event
            
            for t3 in events_by_type[event3]:
                time_diff_2_3 = (t3 - t2).total_seconds() / 60  # Convert to minutes
                if 0 < time_diff_2_3 <= max_time_gap:
                    # This is a minimal occurrence
                    occurrences.append([occ_12[0], occ_12[1], (t3, t3)])
                    
                    # Calculate total time from event1 to event3
                    t1 = occ_12[0][0]  # Start time of first event
                    total_time = (t3 - t1).total_seconds() / 60
                    time_diff_1_2 = (t2 - t1).total_seconds() / 60
                    
                    time_gaps.append((time_diff_1_2, time_diff_2_3, total_time))
        
        if len(occurrences) >= min_support:
            episode_occurrences[candidate] = occurrences
            frequent_episodes[candidate] = occurrences
            
            # Calculate statistics for time gaps
            if time_gaps:
                # For 3-event episodes, we store the total time and individual gaps
                avg_total = sum(g[2] for g in time_gaps) / len(time_gaps)
                min_total = min(g[2] for g in time_gaps)
                max_total = max(g[2] for g in time_gaps)
                
                # Store detailed information
                episode_details[candidate] = {
                    'avg_total_minutes': avg_total,
                    'min_total_minutes': min_total,
                    'max_total_minutes': max_total,
                    'count': len(occurrences),
                    'max_time_gap': max_time_gap,
                    'time_gaps': time_gaps,
                    'consecutive_days': count_consecutive_days(occurrences)
                }
    
    if verbose:
        print(f"Found {sum(1 for ep in frequent_episodes if len(ep) == 3)} frequent 3-event episodes")
    
    # Count occurrences for each frequent episode
    result = {}
    for episode, occurrences in frequent_episodes.items():
        result[episode] = len(occurrences)
    
    # Sort by count
    sorted_episodes = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    if verbose:
        print("\nTop episodes by occurrence count:")
        for i, (episode, count) in enumerate(list(sorted_episodes.items())[:20]):
            if i >= 20:
                break
                
            episode_str = " → ".join(episode)
            
            # Display time gap information if available
            gap_info = ""
            if episode in episode_details and len(episode) == 2:
                stats = episode_details[episode]
                gap_info = f", avg gap: {stats['avg_gap_minutes']:.1f}min (range: {stats['min_gap_minutes']:.1f}-{stats['max_gap_minutes']:.1f}min)"
            elif episode in episode_details and len(episode) == 3:
                stats = episode_details[episode]
                gap_info = f", avg total time: {stats['avg_total_minutes']:.1f}min"
            
            print(f"{i+1}. {episode_str}: {count} occurrences{gap_info}")
    
    return sorted_episodes, episode_details

def detect_episodic_patterns(event_stream, min_minutes=10, max_minutes=60*24, min_occurrences=5, include_triplets=True, verbose=False, simplified_output=True):
    """
    Detect episodic patterns of events that frequently occur together within a specific time range.
    
    This is a custom implementation that focuses on finding pairs and triplets of events that have
    consistent time gaps between them.
    
    Parameters:
        event_stream (list): List of (timestamp, event_name) tuples
        min_minutes (int): Minimum time gap between events to consider
        max_minutes (int): Maximum time gap between events to consider
        min_occurrences (int): Minimum number of occurrences required
        include_triplets (bool): Whether to include 3-event patterns
        verbose (bool): Whether to print detailed information
        simplified_output (bool): Whether to print a simplified summary
        
    Returns:
        tuple: (patterns, simplified_patterns)
            - patterns: List of detailed pattern dictionaries
            - simplified_patterns: List of simplified pattern summaries for easy understanding
    """
    if not event_stream:
        return [], []
    
    if verbose:
        print(f"Detecting episodic patterns with time gap range of {min_minutes}-{max_minutes} minutes")
        print(f"Minimum occurrences threshold: {min_occurrences}")
        print(f"Processing {len(event_stream)} events\n")
    
    # Sort events by time
    event_stream.sort(key=lambda x: x[0])
    
    # Create a dictionary to store event pairs and their time gaps
    pattern_gaps = {}
    
    # Group events by day
    events_by_day = {}
    for event_time, event_name in event_stream:
        day = event_time.date()
        if day not in events_by_day:
            events_by_day[day] = []
        events_by_day[day].append((event_time, event_name))
    
    if verbose:
        print(f"Found events on {len(events_by_day)} different days")
    
    # Total number of days for support calculation
    total_days = len(events_by_day)
    
    # For each day, find patterns
    for day, day_events in events_by_day.items():
        day_events.sort(key=lambda x: x[0])
        
        if verbose and day_events:
            print(f"Day {day}: {len(day_events)} events")
        
        # Check pairs of events
        for i, (time_i, event_i) in enumerate(day_events):
            for j in range(i+1, len(day_events)):
                time_j, event_j = day_events[j]
                
                # Calculate time difference in minutes
                time_diff = (time_j - time_i).total_seconds() / 60
                
                # Check if within desired range
                if min_minutes <= time_diff <= max_minutes:
                    pattern_key = (event_i, event_j)
                    if pattern_key not in pattern_gaps:
                        pattern_gaps[pattern_key] = []
                    
                    pattern_gaps[pattern_key].append({
                        'day': day,
                        'first_time': time_i,
                        'second_time': time_j,
                        'gap_minutes': time_diff
                    })
                    
                    # Check for triplets if enabled
                    if include_triplets:
                        for k in range(j+1, len(day_events)):
                            time_k, event_k = day_events[k]
                            
                            # Calculate time difference between second and third events
                            time_diff_jk = (time_k - time_j).total_seconds() / 60
                            
                            # Check if within desired range
                            if min_minutes <= time_diff_jk <= max_minutes:
                                triplet_key = (event_i, event_j, event_k)
                                if triplet_key not in pattern_gaps:
                                    pattern_gaps[triplet_key] = []
                                
                                pattern_gaps[triplet_key].append({
                                    'day': day,
                                    'first_time': time_i,
                                    'second_time': time_j,
                                    'third_time': time_k,
                                    'gap1_minutes': time_diff,
                                    'gap2_minutes': time_diff_jk,
                                    'total_gap_minutes': time_diff + time_diff_jk
                                })
    
    if verbose:
        print(f"\nFound {len(pattern_gaps)} potential episode patterns")
    
    # Filter patterns by minimum occurrences
    frequent_patterns = []
    
    # Get all unique events for lift calculation
    all_events = set()
    for event_time, event_name in event_stream:
        all_events.add(event_name)
    
    # Calculate event frequencies (for lift calculation)
    event_counts = {}
    for day, day_events in events_by_day.items():
        day_unique_events = set(event for _, event in day_events)
        for event in day_unique_events:
            event_counts[event] = event_counts.get(event, 0) + 1
    
    for pattern, occurrences in pattern_gaps.items():
        if len(occurrences) >= min_occurrences:
            # Create pattern object based on pattern type
            if len(pattern) == 2:  # Pair
                # Calculate statistics
                gaps = [occ['gap_minutes'] for occ in occurrences]
                avg_gap = sum(gaps) / len(gaps)
                gap_std = (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 if len(gaps) > 1 else 0
                
                # Time band calculation (±1 standard deviation around the average)
                lower_band = max(0, avg_gap - gap_std)
                upper_band = avg_gap + gap_std
                
                # Percentage of occurrences within the time band
                in_band_count = sum(1 for g in gaps if lower_band <= g <= upper_band)
                band_percentage = in_band_count / len(gaps) if gaps else 0
                
                # Calculate consistency measure
                consistency = 1 - (gap_std / avg_gap) if avg_gap > 0 else 0
                
                # Count consecutive days
                consecutive_days = count_consecutive_days_from_occurrences(occurrences)
                
                # Calculate support (proportion of days this pattern appears)
                pattern_days = set(occ['day'] for occ in occurrences)
                support = len(pattern_days) / total_days if total_days > 0 else 0
                
                # Calculate lift
                # Lift = P(A,B) / (P(A) * P(B))
                prob_a = event_counts.get(pattern[0], 0) / total_days if total_days > 0 else 0
                prob_b = event_counts.get(pattern[1], 0) / total_days if total_days > 0 else 0
                lift = support / (prob_a * prob_b) if prob_a * prob_b > 0 else 0
                
                pattern_obj = {
                    'pattern_type': 'pair',
                    'first_event': pattern[0],
                    'second_event': pattern[1],
                    'occurrences': len(occurrences),
                    'avg_gap_minutes': avg_gap,
                    'gap_std_minutes': gap_std,
                    'min_gap_minutes': min(gaps),
                    'max_gap_minutes': max(gaps),
                    'lower_band': lower_band,
                    'upper_band': upper_band,
                    'band_percentage': band_percentage,
                    'consistency': consistency,
                    'consecutive_days': consecutive_days,
                    'support': support,
                    'lift': lift,
                    'instances': occurrences
                }
                
                frequent_patterns.append(pattern_obj)
                
            elif len(pattern) == 3 and include_triplets:  # Triplet
                # Calculate statistics for both gaps
                gaps1 = [occ['gap1_minutes'] for occ in occurrences]
                gaps2 = [occ['gap2_minutes'] for occ in occurrences]
                total_gaps = [occ['total_gap_minutes'] for occ in occurrences]
                
                avg_gap1 = sum(gaps1) / len(gaps1)
                gap1_std = (sum((g - avg_gap1) ** 2 for g in gaps1) / len(gaps1)) ** 0.5 if len(gaps1) > 1 else 0
                
                avg_gap2 = sum(gaps2) / len(gaps2)
                gap2_std = (sum((g - avg_gap2) ** 2 for g in gaps2) / len(gaps2)) ** 0.5 if len(gaps2) > 1 else 0
                
                avg_total_gap = sum(total_gaps) / len(total_gaps)
                total_gap_std = (sum((g - avg_total_gap) ** 2 for g in total_gaps) / len(total_gaps)) ** 0.5 if len(total_gaps) > 1 else 0
                
                # Time bands for each gap
                lower_band1 = max(0, avg_gap1 - gap1_std)
                upper_band1 = avg_gap1 + gap1_std
                
                lower_band2 = max(0, avg_gap2 - gap2_std)
                upper_band2 = avg_gap2 + gap2_std
                
                # Count occurrences within the bands
                in_band_count1 = sum(1 for g in gaps1 if lower_band1 <= g <= upper_band1)
                band1_percentage = in_band_count1 / len(gaps1) if gaps1 else 0
                
                in_band_count2 = sum(1 for g in gaps2 if lower_band2 <= g <= upper_band2)
                band2_percentage = in_band_count2 / len(gaps2) if gaps2 else 0
                
                # Calculate consistency measures
                consistency1 = 1 - (gap1_std / avg_gap1) if avg_gap1 > 0 else 0
                consistency2 = 1 - (gap2_std / avg_gap2) if avg_gap2 > 0 else 0
                overall_consistency = (consistency1 + consistency2) / 2
                
                # Count consecutive days
                consecutive_days = count_consecutive_days_from_occurrences(occurrences)
                
                # Calculate support
                pattern_days = set(occ['day'] for occ in occurrences)
                support = len(pattern_days) / total_days if total_days > 0 else 0
                
                # Calculate lift (simplified for triplets)
                prob_a = event_counts.get(pattern[0], 0) / total_days if total_days > 0 else 0
                prob_b = event_counts.get(pattern[1], 0) / total_days if total_days > 0 else 0
                prob_c = event_counts.get(pattern[2], 0) / total_days if total_days > 0 else 0
                lift = support / (prob_a * prob_b * prob_c) if prob_a * prob_b * prob_c > 0 else 0
                
                pattern_obj = {
                    'pattern_type': 'triplet',
                    'first_event': pattern[0],
                    'second_event': pattern[1],
                    'third_event': pattern[2],
                    'occurrences': len(occurrences),
                    'avg_gap1_minutes': avg_gap1,
                    'gap1_std_minutes': gap1_std,
                    'min_gap1_minutes': min(gaps1),
                    'max_gap1_minutes': max(gaps1),
                    'avg_gap2_minutes': avg_gap2,
                    'gap2_std_minutes': gap2_std,
                    'min_gap2_minutes': min(gaps2),
                    'max_gap2_minutes': max(gaps2),
                    'avg_total_gap_minutes': avg_total_gap,
                    'total_gap_std_minutes': total_gap_std,
                    'band1_percentage': band1_percentage,
                    'band2_percentage': band2_percentage,
                    'consistency1': consistency1,
                    'consistency2': consistency2,
                    'overall_consistency': overall_consistency,
                    'consecutive_days': consecutive_days,
                    'support': support,
                    'lift': lift,
                    'instances': occurrences
                }
                
                frequent_patterns.append(pattern_obj)
    
    # Sort by occurrences, consistency, and lift
    sorted_patterns = sorted(frequent_patterns, 
                             key=lambda x: (x['lift'], 
                                           x.get('consistency', x.get('overall_consistency', 0)),
                                           x['occurrences']), 
                             reverse=True)
    
    if verbose:
        print(f"\nFound {len(sorted_patterns)} frequent episodic patterns")
        
        for i, pattern in enumerate(sorted_patterns[:20]):
            if i >= 20:
                break
            
            if pattern['pattern_type'] == 'pair':
                consistency_desc = "very consistent" if pattern['consistency'] > 0.8 else \
                                   "somewhat consistent" if pattern['consistency'] > 0.5 else "variable"
                
                time_band = f"{pattern['lower_band']:.1f}-{pattern['upper_band']:.1f}min ({pattern['band_percentage']*100:.0f}% of occurrences)"
                
                print(f"{i+1}. {pattern['first_event']} → {pattern['second_event']}: "
                      f"{pattern['occurrences']} occurrences, "
                      f"avg gap: {pattern['avg_gap_minutes']:.1f}min (range: {pattern['min_gap_minutes']:.1f}-{pattern['max_gap_minutes']:.1f}min), "
                      f"time band: {time_band}, "
                      f"consistency: {consistency_desc} ({pattern['consistency']:.2f}), "
                      f"support: {pattern['support']:.2f}, lift: {pattern['lift']:.2f}")
            else:  # Triplet
                consistency_desc = "very consistent" if pattern['overall_consistency'] > 0.8 else \
                                   "somewhat consistent" if pattern['overall_consistency'] > 0.5 else "variable"
                
                print(f"{i+1}. {pattern['first_event']} → {pattern['second_event']} → {pattern['third_event']}: "
                      f"{pattern['occurrences']} occurrences, "
                      f"gaps: {pattern['avg_gap1_minutes']:.1f}min + {pattern['avg_gap2_minutes']:.1f}min = {pattern['avg_total_gap_minutes']:.1f}min, "
                      f"consistency: {consistency_desc} ({pattern['overall_consistency']:.2f}), "
                      f"support: {pattern['support']:.2f}, lift: {pattern['lift']:.2f}")
    
    # Create simplified pattern summaries
    simplified_patterns = []
    for pattern in sorted_patterns:
        if pattern['pattern_type'] == 'pair':
            summary = (f"{pattern['first_event']} → {pattern['second_event']}: {pattern['occurrences']} occurrences, "
                     f"avg gap: {pattern['avg_gap_minutes']:.1f}min, consistency: {pattern['consistency']:.2f}, "
                     f"support: {pattern['support']:.2f}, lift: {pattern['lift']:.2f}")
        else:  # Triplet
            summary = (f"{pattern['first_event']} → {pattern['second_event']} → {pattern['third_event']}: "
                     f"{pattern['occurrences']} occurrences, total gap: {pattern['avg_total_gap_minutes']:.1f}min, "
                     f"consistency: {pattern['overall_consistency']:.2f}, support: {pattern['support']:.2f}, lift: {pattern['lift']:.2f}")
        
        simplified_patterns.append(summary)
    
    if simplified_output:
        print("\nSimplified pattern summaries:")
        print("\nConsistency: How regular the time gaps are (1.0 = perfectly consistent, 0.0 = highly variable)")
        print("Support: Proportion of days this pattern appears (higher = more common)")
        print("Lift: How much more likely events occur together vs. by chance (higher = stronger association)")
        print("Time band: Range where most occurrences fall (± 1 standard deviation around average)")
        print()
        
        for i, pattern_summary in enumerate(simplified_patterns[:20]):
            print(f"{i+1}. {pattern_summary}")
    
    return sorted_patterns, simplified_patterns

def count_consecutive_days(occurrences):
    """Count the maximum number of consecutive days where this pattern appears"""
    if not occurrences or not occurrences[0]:
        return 0
    
    # Extract dates from occurrences
    dates = []
    for occ in occurrences:
        if occ and occ[0] and len(occ[0]) > 0:
            # Get the first timestamp from the first event
            dates.append(occ[0][0].date())
    
    return count_consecutive_days_from_dates(dates)

def count_consecutive_days_from_occurrences(occurrences):
    """Count the maximum number of consecutive days from occurrence objects"""
    if not occurrences:
        return 0
    
    # Extract dates from occurrences
    dates = [occ['day'] for occ in occurrences]
    
    return count_consecutive_days_from_dates(dates)

def count_consecutive_days_from_dates(dates):
    """Count the maximum number of consecutive days from a list of dates"""
    if not dates:
        return 0
    
    # Remove duplicates and sort
    unique_dates = sorted(set(dates))
    
    if len(unique_dates) == 1:
        return 1
    
    # Find longest streak
    max_streak = current_streak = 1
    for i in range(1, len(unique_dates)):
        # Check if consecutive
        if (unique_dates[i] - unique_dates[i-1]).days == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    
    return max_streak

def explain_minepi_results(sorted_episodes, episode_details, max_time_gap, top_n=10):
    """
    Explain MINEPI results in detail with a focus on interpretation.
    
    Parameters:
        sorted_episodes (dict): Dictionary of episodes with their occurrence counts
        episode_details (dict): Dictionary with detailed information about each episode
        max_time_gap (int): Maximum time gap used in MINEPI (minutes)
        top_n (int): Number of top episodes to explain
        
    Returns:
        None (prints explanation)
    """
    print("\n=== MINEPI RESULTS EXPLANATION ===\n")
    print(f"Maximum time gap used: {max_time_gap} minutes")
    print("Occurrence count meaning: The number of times this sequence appears")
    print("Example: An occurrence count of 7 means this sequence occurred 7 times\n")
    
    print("Top episodes by occurrence count:")
    
    for i, (episode, count) in enumerate(list(sorted_episodes.items())[:top_n]):
        if i >= top_n:
            break
            
        episode_str = " → ".join(episode)
        
        print(f"{i+1}. {episode_str}: {count} occurrences")
        
        # Interpret the results
        if len(episode) == 1:
            print(f"   Interpretation: The event '{episode[0]}' occurred {count} times.")
        elif len(episode) == 2:
            print(f"   Interpretation: The sequence '{episode_str}' occurred {count} times.")
            
            # Add time gap information if available
            if episode in episode_details:
                stats = episode_details[episode]
                print(f"   Time relationship: On average, {episode[1]} occurs {stats['avg_gap_minutes']:.1f} minutes after {episode[0]}")
                print(f"   Gap range: {stats['min_gap_minutes']:.1f}-{stats['max_gap_minutes']:.1f} minutes")
                
                consistency = 1 - (stats['std_gap_minutes'] / stats['avg_gap_minutes']) if stats['avg_gap_minutes'] > 0 else 0
                consistency_desc = "very consistent" if consistency > 0.8 else "somewhat consistent" if consistency > 0.5 else "variable"
                print(f"   Consistency: {consistency_desc} ({consistency:.2f})")
            
            # Add consecutive days information if available
            if episode in episode_details and 'consecutive_days' in episode_details[episode]:
                consecutive_days = episode_details[episode]['consecutive_days']
                print(f"   Occurs on {consecutive_days} consecutive days")
        
        print()
    
    print("\nUnderstanding the numbers:")
    print(f"1. MINEPI finds minimal occurrences of episodes within a time window.")
    print("2. The occurrence count shows how many times the pattern appears.")
    print("3. Higher occurrence count = more frequent pattern.")
    print("4. Events in a sequence occur in order, with a maximum time gap between them.")
    print("5. A → B doesn't necessarily imply causation, just temporal sequence.")

def count_consecutive_days_from_dates(dates):
    """Count the maximum number of consecutive days from a list of dates"""
    if not dates:
        return 0
    
    # Remove duplicates and sort
    unique_dates = sorted(set(dates))
    
    if len(unique_dates) == 1:
        return 1
    
    # Find longest streak
    max_streak = current_streak = 1
    for i in range(1, len(unique_dates)):
        # Check if consecutive
        if (unique_dates[i] - unique_dates[i-1]).days == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    
    return max_streak

def explain_minepi_results(sorted_episodes, episode_details, max_time_gap, top_n=10):
    """
    Explain MINEPI results in detail with a focus on interpretation.
    
    Parameters:
        sorted_episodes (dict): Dictionary of episodes with their occurrence counts
        episode_details (dict): Dictionary with detailed information about each episode
        max_time_gap (int): Maximum time gap used in MINEPI (minutes)
        top_n (int): Number of top episodes to explain
        
    Returns:
        None (prints explanation)
    """
    print("\n=== MINEPI RESULTS EXPLANATION ===\n")
    print(f"Maximum time gap used: {max_time_gap} minutes")
    print("Occurrence count meaning: The number of times this sequence appears")
    print("Example: An occurrence count of 7 means this sequence occurred 7 times\n")
    
    print("Top episodes by occurrence count:")
    
    for i, (episode, count) in enumerate(list(sorted_episodes.items())[:top_n]):
        if i >= top_n:
            break
            
        episode_str = " → ".join(episode)
        
        print(f"{i+1}. {episode_str}: {count} occurrences")
        
        # Interpret the results
        if len(episode) == 1:
            print(f"   Interpretation: The event '{episode[0]}' occurred {count} times.")
        elif len(episode) == 2:
            print(f"   Interpretation: The sequence '{episode_str}' occurred {count} times.")
            
            # Add time gap information if available
            if episode in episode_details:
                stats = episode_details[episode]
                print(f"   Time relationship: On average, {episode[1]} occurs {stats['avg_gap_minutes']:.1f} minutes after {episode[0]}")
                print(f"   Gap range: {stats['min_gap_minutes']:.1f}-{stats['max_gap_minutes']:.1f} minutes")
                
                consistency = 1 - (stats['std_gap_minutes'] / stats['avg_gap_minutes']) if stats['avg_gap_minutes'] > 0 else 0
                consistency_desc = "very consistent" if consistency > 0.8 else "somewhat consistent" if consistency > 0.5 else "variable"
                print(f"   Consistency: {consistency_desc} ({consistency:.2f})")
            
            # Add consecutive days information if available
            if episode in episode_details and 'consecutive_days' in episode_details[episode]:
                consecutive_days = episode_details[episode]['consecutive_days']
                print(f"   Occurs on {consecutive_days} consecutive days")
        
        print()
    
    print("\nUnderstanding the numbers:")
    print(f"1. MINEPI finds minimal occurrences of episodes within a time window.")
    print("2. The occurrence count shows how many times the pattern appears.")
    print("3. Higher occurrence count = more frequent pattern.")
    print("4. Events in a sequence occur in order, with a maximum time gap between them.")
    print("5. A → B doesn't necessarily imply causation, just temporal sequence.")


import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, PC
# from pgmpy.estimators import BicScore
# from pgmpy.estimators.scores import BicScore
from pgmpy.estimators import BDeu, K2, BIC
import networkx as nx
import matplotlib.pyplot as plt


def bayesian_network(df, scoring_method='bic', highlight_nodes=None, target_focus=None):
    """
    Create and visualize a Bayesian network from data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to learn the Bayesian network from
    scoring_method : str, default='bic'
        The scoring method to use ('bic', 'k2', or 'bdeu')
    highlight_nodes : list, default=None
        Optional list of node names to highlight in a different color
    target_focus : str, default=None
        Optional mode to focus on highlighted nodes: 'removes' or 'grays' nodes not connected to highlights
        
    Returns:
    --------
    tuple:
        - model: The learned Bayesian network model
        - connections_df: DataFrame with details about all connections found
    """
    # Option 1: Score-based learning with Hill Climbing

    # df = df.copy()
    # df = df.replace(0, np.nan)

    hc = HillClimbSearch(df)
    if scoring_method == 'bic':
        score = BIC(df)
    elif scoring_method == 'k2':
        score = K2(df)
    elif scoring_method == 'bdeu':
        score = BDeu(df)

    # set any 0 values to NaN

    # constraints = [('shower:last', 'LEP:datetime')]

    best_model = hc.estimate(scoring_method=score)

    # pc = PC(df)
    # best_model = pc.estimate(significance_level=0.05)

    
    # Identify root nodes (nodes with no parents)
    root_nodes = {node for node in best_model.nodes() if not list(best_model.predecessors(node))}
    
    # Calculate prior distributions for root nodes
    prior_distributions = {}
    for node in root_nodes:
        if node in df.columns:
            # For categorical variables
            if df[node].dtype == 'object' or df[node].dtype == 'category':
                prior_distributions[node] = df[node].value_counts(normalize=True).to_dict()
            # For numerical variables, compute basic statistics
            else:
                prior_distributions[node] = {
                    'mean': df[node].mean(),
                    'median': df[node].median(),
                    'std': df[node].std(),
                    'min': df[node].min(),
                    'max': df[node].max()
                }
    
    # Create a DataFrame of all connections
    connections = []
    
    for parent, child in best_model.edges():
        # Calculate edge strength metrics
        correlation = None
        mutual_info = None
        association_count = None
        counter_example = None
        
        # For numerical data, calculate correlation
        if (parent in df.columns and child in df.columns and
            df[parent].dtype in ['float64', 'int64'] and 
            df[child].dtype in ['float64', 'int64']):
            
            correlation = df[parent].corr(df[child])
            
            # Find counter-example for strong correlations (above 0.9)
            if abs(correlation) > 0.9:
                # Calculate z-scores for both variables
                parent_z = (df[parent] - df[parent].mean()) / df[parent].std()
                child_z = (df[child] - df[child].mean()) / df[child].std()
                
                # Look for cases where the signs differ (negative correlation)
                # or where one is high but the other isn't (positive correlation)
                if correlation > 0:
                    # For positive correlation, find cases where one is high but other isn't
                    diff = abs(parent_z - child_z)
                    counter_idx = diff.nlargest(1).index[0] if not diff.empty else None
                else:
                    # For negative correlation, find cases where they go in same direction
                    product = parent_z * child_z
                    counter_idx = product.nlargest(1).index[0] if not product.empty else None
                
                if counter_idx is not None:
                    counter_example = {
                        'index': counter_idx,
                        parent: df.loc[counter_idx, parent],
                        child: df.loc[counter_idx, child]
                    }
            
        # For categorical data, calculate association counts
        elif parent in df.columns and child in df.columns:
            counts = df.groupby([parent, child]).size().reset_index(name='count')
            association_count = counts['count'].sum()
        
        # Add to connections list
        connections.append({
            'from': parent,
            'to': child,
            'correlation': correlation,
            'association_count': association_count,
            'counter_example': counter_example
        })
    
    # Create DataFrame of connections
    connections_df = pd.DataFrame(connections)
    
    # Visualize the resulting network
    def plot_network(model, title):
        G = nx.DiGraph()
        G.add_edges_from(model.edges())
        
        # Find relevant nodes when using target_focus
        relevant_nodes = set()
        
        # Handle target focus mode
        if target_focus and highlight_nodes:
            for target_node in highlight_nodes:
                if target_node not in G.nodes():
                    print(f"Warning: Node '{target_node}' not found in graph.")
                    continue
                
                # Helper function to find all ancestors recursively
                def find_ancestors(node, visited=None):
                    if visited is None:
                        visited = set()
                    visited.add(node)
                    relevant_nodes.add(node)  # Add to relevant nodes
                    for pred in G.predecessors(node):
                        relevant_nodes.add(pred)  # Add predecessor
                        if pred not in visited:
                            find_ancestors(pred, visited)
                
                # Add target node and find its ancestors
                find_ancestors(target_node)
            
            # For 'removes' mode, remove unrelated nodes
            if target_focus == 'removes':
                nodes_to_remove = [n for n in G.nodes() if n not in relevant_nodes]
                for node in nodes_to_remove:
                    G.remove_node(node)
        
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(G)
        
        # Default node color is lightblue with transparency
        node_colors = []
        node_alpha = []
        
        for node in G.nodes():
            # Default color and alpha
            color = 'lightblue'
            alpha = 0.7
            
            # If root node, use a different color
            if node in root_nodes:
                color = 'lightgreen'
            
            # If in target_focus 'grays' mode, gray out unrelated nodes
            if (target_focus == 'grays' and highlight_nodes and 
                node not in relevant_nodes):
                color = 'lightgray'
                alpha = 0.3
                
            # If node should be highlighted, change its color
            if highlight_nodes and node in highlight_nodes:
                color = 'red'
                
            node_colors.append(color)
            node_alpha.append(alpha)
        
        # Draw nodes with transparency
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=1500,
                              alpha=node_alpha)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Create edge color mapping based on correlation or association strength
        edge_cmap = plt.cm.YlOrRd  # Yellow-to-Red colormap for strength
        edge_colors = []
        edge_widths = []
        edge_alpha = []
        
        # Find max values for normalization
        max_corr = 0
        max_assoc = 0
        
        for i, (u, v) in enumerate(G.edges()):
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                if corr is not None and not pd.isna(corr) and abs(corr) > max_corr:
                    max_corr = abs(corr)
                    
                if assoc is not None and not pd.isna(assoc) and assoc > max_assoc:
                    max_assoc = assoc
        
        # Normalize values and set edge properties
        for i, (u, v) in enumerate(G.edges()):
            # Default edge settings
            color = 'lightgray'
            width = 1.0
            alpha = 1.0
            
            # Get edge data from connections DataFrame
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                # Use correlation or association count for color intensity
                if corr is not None and not pd.isna(corr):
                    # Normalize correlation between 0 and 1 for color mapping
                    color_val = abs(corr) / max(max_corr, 0.01)  # Avoid division by zero
                    color = edge_cmap(color_val)
                    width = 1 + 2 * color_val  # Width between 1 and 3
                elif assoc is not None and not pd.isna(assoc):
                    # Normalize association count
                    color_val = assoc / max(max_assoc, 1)  # Avoid division by zero
                    color = edge_cmap(color_val)
                    width = 1 + 2 * color_val  # Width between 1 and 3
            
            # In 'grays' mode, check if either end is unrelated to target
            if (target_focus == 'grays' and highlight_nodes and 
                (u not in relevant_nodes or v not in relevant_nodes)):
                color = 'lightgray'
                alpha = 0.3
                width = 1.0
                
            edge_colors.append(color)
            edge_widths.append(width)
            edge_alpha.append(alpha)
        
        # Draw edges with varying width and color based on strength
        nx.draw_networkx_edges(G, pos, 
                             arrowsize=20, 
                             edge_color=edge_colors,
                             width=edge_widths,
                             alpha=edge_alpha)
        
        # Calculate edge weights based on mutual information or correlation
        edge_labels = {}
        for u, v in G.edges():
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                if corr is not None and not pd.isna(corr):
                    edge_labels[(u, v)] = f"{corr:.2f}"
                elif assoc is not None and not pd.isna(assoc):
                    edge_labels[(u, v)] = f"{assoc}"
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Create legend patches
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        
        legend_patches = []
        
        # Regular node legend
        if 'lightblue' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightblue', alpha=0.7, label='Regular Nodes'))
        
        # Root node legend
        if 'lightgreen' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightgreen', alpha=0.7, label='Root Nodes'))
        
        # Highlighted node legend
        if highlight_nodes and 'red' in node_colors:
            legend_patches.append(mpatches.Patch(color='red', alpha=0.7, label='Highlighted Nodes'))
        
        # Gray node legend (for target_focus='grays')
        if target_focus == 'grays' and 'lightgray' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightgray', alpha=0.3, label='Unrelated Nodes'))
        
        # Edge strength legend
        if max_corr > 0 or max_assoc > 0:
            # Create gradient legend for edge strength
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            
            # Add colorbar for edge strength
            ax2 = plt.axes([0.92, 0.1, 0.02, 0.3])  # Position colorbar
            ax2.imshow(gradient.T, aspect='auto', cmap=edge_cmap)
            ax2.set_title('Edge\nStrength')
            ax2.set_xticks([])
            ax2.set_yticks([0, 255])
            ax2.set_yticklabels(['Low', 'High'])
        
        # Add legend if we have any patches
        if legend_patches:
            plt.legend(handles=legend_patches, loc='upper left')
            
        # Set title based on modes
        if target_focus and highlight_nodes:
            title += f" (Focus on highlighted nodes)"
            
        plt.title(title)
        plt.axis('off')  # Turn off axis
        plt.show()

    plot_network(best_model, "Bayesian Network from Hill Climbing with " + scoring_method)

    # Analyze direct and indirect relationships
    lep_time_deps = [parent for parent, child in best_model.edges() if child == 'LEP_time']
    print("Direct dependencies of LEP_time:", lep_time_deps)

    # Find Markov Blanket of LEP_time
    def find_markov_blanket(model, node):
        blanket = set()
        for parent, child in model.edges():
            if child == node:
                blanket.add(parent)
            elif parent == node:
                blanket.add(child)
                # Add other parents of this child
                for p, c in model.edges():
                    if c == child and p != node:
                        blanket.add(p)
        return blanket

    # Show Markov blanket for each highlighted node
    if highlight_nodes:
        for node in highlight_nodes:
            if node in best_model.nodes():
                print(f"Markov Blanket of {node}:", find_markov_blanket(best_model, node))
    
    # Add Markov blanket information to the connections DataFrame
    if not connections_df.empty:
        # Add a column for highlight status
        connections_df['is_highlight_source'] = connections_df['from'].isin(highlight_nodes) if highlight_nodes else False
        connections_df['is_highlight_target'] = connections_df['to'].isin(highlight_nodes) if highlight_nodes else False
        
        # For each highlighted node, add columns indicating if the edge is in its Markov blanket
        if highlight_nodes:
            for node in highlight_nodes:
                if node in best_model.nodes():
                    blanket = find_markov_blanket(best_model, node)
                    connections_df[f'in_{node}_markov_blanket'] = connections_df.apply(
                        lambda row: row['from'] in blanket or row['to'] in blanket, 
                        axis=1
                    )
    
    # Add information about prior distributions for root nodes
    root_node_info = pd.DataFrame([
        {'node': node, 'prior_distribution': str(prior_distributions.get(node, {}))}
        for node in root_nodes
    ])
    
    return best_model, connections_df, root_node_info