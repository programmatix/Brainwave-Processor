import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def autoencoder_random_forests_pytorch(df, target_col=None, is_classifier=False, 
                                      autoencoder_dims=None, random_forest_params=None,
                                      test_size=0.2, random_state=42, 
                                      learning_rate=0.001, batch_size=32, epochs=100):
    """
    Implements Autoencoder Random Forests using PyTorch for feature extraction and prediction.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and targets
        target_col (str, optional): Name of the target column. If None, only performs feature extraction
        is_classifier (bool): Whether this is a classification (True) or regression (False) task
        autoencoder_dims (list, optional): Dimensions for autoencoder hidden layers. Default: [df.shape[1]//2]
        random_forest_params (dict, optional): Parameters for the Random Forest model
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        learning_rate (float): Learning rate for the autoencoder optimizer
        batch_size (int): Batch size for training
        epochs (int): Maximum number of training epochs
        
    Returns:
        dict: Results containing the trained autoencoder, trained RF model, and evaluation metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Make a copy of dataframe to avoid modifying the original
    df = df.copy()
    
    # Default autoencoder dimensions if not provided
    if autoencoder_dims is None:
        autoencoder_dims = [df.shape[1] // 2]
    
    # Default random forest parameters if not provided
    if random_forest_params is None:
        random_forest_params = {
            'n_estimators': 100,
            'max_depth': 8,
            'random_state': random_state
        }
    
    # Prepare X and y
    if target_col is not None and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build PyTorch autoencoder
    input_dim = X_scaled.shape[1]
    bottleneck_dim = max(1, autoencoder_dims[-1] // 2)
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dims, bottleneck_dim):
            super(Autoencoder, self).__init__()
            
            # Encoder layers
            encoder_layers = []
            encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
            encoder_layers.append(nn.ReLU())
            
            for i in range(len(hidden_dims)-1):
                encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                encoder_layers.append(nn.ReLU())
            
            encoder_layers.append(nn.Linear(hidden_dims[-1], bottleneck_dim))
            encoder_layers.append(nn.ReLU())
            
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Decoder layers
            decoder_layers = []
            decoder_layers.append(nn.Linear(bottleneck_dim, hidden_dims[-1]))
            decoder_layers.append(nn.ReLU())
            
            for i in range(len(hidden_dims)-1, 0, -1):
                decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
            
            decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
            
            self.decoder = nn.Sequential(*decoder_layers)
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
        def encode(self, x):
            return self.encoder(x)
    
    # Initialize model
    model = Autoencoder(input_dim, autoencoder_dims, bottleneck_dim)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if y is not None:
        # Split the data if target is provided
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Create DataLoader for batching
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, _ in train_loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, X_test_tensor)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            model.train()
        
        # Extract features using the encoder
        model.eval()
        with torch.no_grad():
            X_train_encoded = model.encode(X_train_tensor).cpu().numpy()
            X_test_encoded = model.encode(X_test_tensor).cpu().numpy()
        
        # Choose the appropriate Random Forest model
        if is_classifier:
            rf_model = RandomForestClassifier(**random_forest_params)
        else:
            rf_model = RandomForestRegressor(**random_forest_params)
        
        # Train Random Forest on encoded features
        rf_model.fit(X_train_encoded, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_encoded)
        
        # Calculate metrics
        if is_classifier:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1
            }
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            metrics = {
                'rmse': rmse,
                'r2_score': r2
            }
        
        # Feature importances
        feature_importances = pd.DataFrame({
            'Feature': [f'Encoded_{i}' for i in range(bottleneck_dim)],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Encode the full dataset
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            X_encoded_full = model.encode(X_tensor).cpu().numpy()
        
        return {
            'autoencoder': model,
            'rf_model': rf_model,
            'metrics': metrics,
            'feature_importances': feature_importances,
            'X_encoded': X_encoded_full
        }
    
    else:
        # Just train the autoencoder for feature extraction if no target is provided
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, _ in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
        
        # Extract features using the encoder
        model.eval()
        with torch.no_grad():
            X_encoded = model.encode(X_tensor).cpu().numpy()
            reconstructed = model(X_tensor).cpu().numpy()
            reconstruction_error = np.mean((reconstructed - X_scaled) ** 2)
        
        return {
            'autoencoder': model,
            'X_encoded': X_encoded,
            'reconstruction_error': reconstruction_error
        }

def tree_based_autoencoder_random_forests(df, target_col=None, is_classifier=False,
                                         n_estimators=100, max_depth=5, n_components=None,
                                         test_size=0.2, random_state=42):
    """
    Implements Autoencoder Random Forests using tree-based encoding and decoding.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and targets
        target_col (str, optional): Name of the target column. If None, only performs feature extraction
        is_classifier (bool): Whether this is a classification (True) or regression (False) task
        n_estimators (int): Number of trees in the encoder and decoder forests
        max_depth (int): Maximum depth of trees
        n_components (int, optional): Number of components in the encoded representation
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Results containing the trained encoder/decoder forests, train RF model, and metrics
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Make a copy of dataframe to avoid modifying the original
    df = df.copy()
    
    # Prepare X and y
    if target_col is not None and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    input_dim = X_scaled.shape[1]
    
    # Determine the number of components for the encoded representation
    if n_components is None:
        n_components = max(1, input_dim // 2)
    n_components = min(n_components, input_dim)  # Cannot have more components than features
    
    # Tree-based encoder - uses a random forest to map original features to a compressed representation
    # Each tree in the encoder forest maps to one component in the encoded space
    encoder_forests = []
    
    for i in range(n_components):
        # Create a random forest regressor for this component
        encoder = RandomForestRegressor(
            n_estimators=max(1, n_estimators // n_components),
            max_depth=max_depth,
            random_state=random_state + i
        )
        # Use a random subset of the features to create diversity
        feature_subset = np.random.choice(
            input_dim, 
            size=max(1, input_dim // 2), 
            replace=False
        )
        # Target for this encoder is a random linear combination of features
        weights = np.random.randn(input_dim)
        target = X_scaled @ weights
        # Fit the encoder
        encoder.fit(X_scaled, target)
        encoder_forests.append(encoder)
    
    # Function to encode data
    def encode(X_data):
        encoded = np.zeros((X_data.shape[0], n_components))
        for i, encoder in enumerate(encoder_forests):
            encoded[:, i] = encoder.predict(X_data)
        return encoded
    
    # Tree-based decoder - uses a random forest to reconstruct original features from encoded representation
    decoder_forests = []
    
    # Encode the data to train decoders
    X_encoded = encode(X_scaled)
    
    # Train a decoder for each original feature
    for i in range(input_dim):
        decoder = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + i + n_components
        )
        # Target is the original feature value
        decoder.fit(X_encoded, X_scaled[:, i])
        decoder_forests.append(decoder)
    
    # Function to decode data
    def decode(encoded_data):
        decoded = np.zeros((encoded_data.shape[0], input_dim))
        for i, decoder in enumerate(decoder_forests):
            decoded[:, i] = decoder.predict(encoded_data)
        return decoded
    
    # Calculate reconstruction error
    X_reconstructed = decode(X_encoded)
    reconstruction_error = np.mean((X_reconstructed - X_scaled) ** 2)
    
    if y is not None:
        # Split the data if target is provided
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Encode the training and test data
        X_train_encoded = encode(X_train)
        X_test_encoded = encode(X_test)
        
        # Choose the appropriate Random Forest model for prediction
        if is_classifier:
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        
        # Train Random Forest on encoded features
        rf_model.fit(X_train_encoded, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_encoded)
        
        # Calculate metrics
        if is_classifier:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1
            }
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            metrics = {
                'rmse': rmse,
                'r2_score': r2
            }
        
        # Calculate feature importances for the encoded features
        feature_importances = pd.DataFrame({
            'Feature': [f'Encoded_{i}' for i in range(n_components)],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return {
            'encoder_forests': encoder_forests,
            'decoder_forests': decoder_forests,
            'rf_model': rf_model,
            'metrics': metrics,
            'feature_importances': feature_importances,
            'X_encoded': X_encoded,
            'reconstruction_error': reconstruction_error,
            'encode_function': encode,
            'decode_function': decode
        }
    
    else:
        # Just return the encoder/decoder if no target is provided
        return {
            'encoder_forests': encoder_forests,
            'decoder_forests': decoder_forests,
            'X_encoded': X_encoded,
            'reconstruction_error': reconstruction_error,
            'encode_function': encode,
            'decode_function': decode
        }