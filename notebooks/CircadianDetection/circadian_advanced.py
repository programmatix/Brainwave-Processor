import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import wittgenstein as wt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def apply_ripper_to_circadian(df_lep):
    """
    Apply RIPPER algorithm to discover rules in circadian data.
    """
    # Prepare data
    # First ensure the data is sorted by date
    if 'dayAndNightOf' in df_lep.columns:
        df_lep = df_lep.sort_values('dayAndNightOf')
    
    # Create target variable - LEP shift in minutes
    df_lep['LEP_shift'] = df_lep['circadian:basic:entries:LEP:datetime'].diff() / 60
    
    # Drop first row with NaN shift and any other NaNs
    df_clean = df_lep.dropna(subset=['LEP_shift'])
    
    # Prepare features - select numeric features and rename for clarity
    feature_mapping = {
        'sunExposureCombined:sunlightBeforeMidday': 'morning_sun_secs',
        'events:luminette:duration': 'luminette_secs',
        'events:shower:last': 'shower_time_ssm',
        'events:luminette:first': 'luminette_time_ssm',
        'sunExposureCombined:sunlightWithin2HoursOfWake': 'early_sun_secs',
        'sunExposureCombined:totalTimeAnySun': 'total_sun_secs',
        'events:shower:count': 'shower_count'
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
    df_lep['LEP_shift'] = df_lep['circadian:basic:entries:LEP:datetime'].diff() / 60
    
    # Drop first row with NaN shift and any other NaNs
    df_clean = df_lep.dropna(subset=['LEP_shift'])
    
    # Prepare features - select numeric features and rename for clarity
    feature_mapping = {
        'sunExposureCombined:sunlightBeforeMidday': 'morning_sunlight',
        'events:luminette:duration': 'luminette_duration',
        'events:shower:last': 'shower_time',
        'events:luminette:first': 'luminette_time',
        'sunExposureCombined:sunlightWithin2HoursOfWake': 'early_sunlight',
        'sunExposureCombined:totalTimeAnySun': 'total_sunlight',
        'events:shower:count': 'shower_count'
    }
    
    # Create simplified feature set
    X = df_clean.select_dtypes(include=['float64', 'int64']).copy()
    
    # Remove target from features if present
    features_to_drop = ['circadian:basic:entries:LEP:datetime', 'LEP_shift']
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
    
    y = df_lep['circadian:basic:entries:LEP:datetime'].diff() / 60  # Convert seconds to minutes
    y = y.fillna(0)  # Fill first day's diff
    
    # Remove the target from features if it exists
    if 'circadian:basic:entries:LEP:datetime' in X.columns:
        X = X.drop(columns=['circadian:basic:entries:LEP:datetime'])
    
    # Create meaningful feature names by simplifying the long column names
    feature_names = {
        'sunExposureCombined:sunlightBeforeMidday': 'morning_sunlight',
        'events:luminette:duration': 'luminette_duration',
        'events:shower:last': 'shower_time',
        'sunExposureCombined:sunlightWithin2HoursOfWake': 'early_sunlight',
        'events:luminette:first': 'luminette_time'
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
    
    print("Feature Importances for LEP Shift Prediction:")
    print(importance_df)
    
    # Build a simple decision tree for rule extraction
    # Use the most important features from the random forest
    top_features = importance_df.head(5)['Feature'].tolist()
    
    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    dt.fit(X_train[top_features], y_train)
    
    # Extract rules as text
    tree_rules = export_text(dt, feature_names=top_features)
    
    print("\nDecision Tree Rules for LEP Shift Prediction:")
    print(tree_rules)
    
    # Create human-readable rules by applying thresholds to the most important features
    print("\nSimplified Human-Readable Rules:")
    
    if 'morning_sunlight' in top_features:
        threshold = np.percentile(X_train['morning_sunlight'], 75)
        effect = y_train[X_train['morning_sunlight'] > threshold].mean() - y_train.mean()
        print(f"- If morning sunlight > {threshold/60:.1f} minutes: LEP shifts by {effect:.1f} minutes")
    
    if 'luminette_duration' in top_features:
        threshold = 900  # 15 minutes
        effect = y_train[X_train['luminette_duration'] > threshold].mean() - y_train.mean()
        print(f"- If Luminette duration > {threshold/60:.1f} minutes: LEP shifts by {effect:.1f} minutes")
    
    if 'shower_time' in top_features:
        early_threshold = 32400  # 9 AM in seconds since midnight
        effect = y_train[X_train['shower_time'] < early_threshold].mean() - y_train.mean()
        print(f"- If shower before {early_threshold/3600:.1f} AM: LEP shifts by {effect:.1f} minutes")
    
    return importance_df, tree_rules


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Prepare data for rule mining by discretizing continuous variables
def prepare_for_rule_mining(df):
    df_discrete = df.copy()
    
    # Discretize your target variable (LEP shift)
    # First create a shift variable (today's LEP compared to yesterday)
    df_discrete['LEP_shift'] = df_discrete['circadian:basic:entries:LEP:datetime'].diff()
    
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
        df_discrete['sunExposureCombined:sunlightBeforeMidday'], 
        bins=[0, 600, 1800, np.inf],  # 0, 10min, 30min, more
        labels=['none', 'moderate', 'substantial']
    )
    
    # Luminette usage
    df_discrete['luminette_usage'] = pd.cut(
        df_discrete['events:luminette:duration'], 
        bins=[0, 1, 900, 1800, np.inf],  # none, 0-15min, 15-30min, more
        labels=['none', 'short', 'standard', 'extended']
    )
    
    # Shower timing relative to wake
    df_discrete['shower_timing'] = pd.cut(
        df_discrete['events:shower:last'], 
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

