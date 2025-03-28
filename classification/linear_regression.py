import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_linear_regression_model(data, target_column, feature_columns, 
                                 variant='standard', alpha=1.0, l1_ratio=0.5,
                                 auto_optimize=False, test_size=0.25, random_state=None):
    """
    Train a Linear Regression model on the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for training
    target_column : str
        The name of the target column
    feature_columns : list
        List of feature column names
    variant : str, default='standard'
        Type of regression: 'standard', 'ridge', 'lasso', or 'elasticnet'
    alpha : float, default=1.0
        Regularization strength (for Ridge, Lasso, and ElasticNet)
    l1_ratio : float, default=0.5
        The mixing parameter for ElasticNet (0 <= l1_ratio <= 1)
    auto_optimize : bool, default=False
        Whether to use GridSearchCV to find optimal parameters
    test_size : float, default=0.25
        The proportion of the dataset to include in the test split
    random_state : int, default=None
        Controls the randomness in the estimator
        
    Returns:
    --------
    tuple
        (model, metrics, X_train, X_test, y_train, y_test)
    """
    # Extract features and target
    X = data[feature_columns].copy()
    y = data[target_column].copy()
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Generate column indices for preprocessor
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
    numeric_indices = [X.columns.get_loc(col) for col in numeric_cols]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Choose regression variant
    if variant == 'standard':
        regressor = LinearRegression()
    elif variant == 'ridge':
        regressor = Ridge(alpha=alpha, random_state=random_state)
    elif variant == 'lasso':
        regressor = Lasso(alpha=alpha, random_state=random_state)
    elif variant == 'elasticnet':
        regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    else:
        raise ValueError(f"Unknown regression variant: {variant}. Choose from 'standard', 'ridge', 'lasso', or 'elasticnet'.")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Auto-optimize if requested
    if auto_optimize:
        param_grid = {}
        
        if variant == 'standard':
            # Standard linear regression has no hyperparameters to tune
            pass
        elif variant == 'ridge':
            param_grid = {
                'regressor__alpha': np.logspace(-3, 3, 7)
            }
        elif variant == 'lasso':
            param_grid = {
                'regressor__alpha': np.logspace(-3, 3, 7)
            }
        elif variant == 'elasticnet':
            param_grid = {
                'regressor__alpha': np.logspace(-3, 3, 7),
                'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        
        if param_grid:  # Only do grid search if there are parameters to tune
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model
            pipeline = grid_search.best_estimator_
        else:
            # Train with default parameters
            pipeline.fit(X_train, y_train)
    else:
        # Train with provided parameters
        pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # Store feature info in the model for later use
    pipeline.feature_columns = feature_columns
    pipeline.target_column = target_column
    
    # If we're using the standard linear regression, expose coefficients for easier access
    if variant == 'standard':
        pipeline.coef_ = pipeline.named_steps['regressor'].coef_
        pipeline.intercept_ = pipeline.named_steps['regressor'].intercept_
    
    # Return model, metrics, and data
    return pipeline, metrics, X_train, X_test, y_train, y_test

def interactive_linear_regression(data):
    """
    Interactive function to let user choose features and target for linear regression.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for training
        
    Returns:
    --------
    tuple
        (model, metrics, X_train, X_test, y_train, y_test)
    """
    # Display available columns
    print("Available columns:")
    for i, col in enumerate(data.columns):
        print(f"{i+1}. {col}")
    
    # Let user select target column
    while True:
        try:
            target_idx = int(input("\nSelect target column number: ")) - 1
            if 0 <= target_idx < len(data.columns):
                target_column = data.columns[target_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Let user select feature columns
    print("\nSelect feature columns (comma-separated numbers, e.g., 1,3,4):")
    while True:
        try:
            feature_indices = [int(x.strip()) - 1 for x in input().split(',')]
            if all(0 <= idx < len(data.columns) for idx in feature_indices) and len(feature_indices) > 0:
                # Ensure target column is not in features
                if target_idx in feature_indices:
                    feature_indices.remove(target_idx)
                    print(f"Removed target column from features.")
                    
                if not feature_indices:
                    print("No valid feature columns selected. Please try again.")
                    continue
                    
                feature_columns = [data.columns[idx] for idx in feature_indices]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter valid numbers separated by commas.")
    
    # Let user choose regression variant
    print("\nChoose regression variant:")
    print("1. Standard Linear Regression")
    print("2. Ridge Regression")
    print("3. Lasso Regression")
    print("4. ElasticNet Regression")
    
    while True:
        try:
            variant_choice = int(input("Enter your choice (1-4): "))
            if 1 <= variant_choice <= 4:
                variant_map = {1: 'standard', 2: 'ridge', 3: 'lasso', 4: 'elasticnet'}
                variant = variant_map[variant_choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Let user decide on hyperparameter optimization
    auto_optimize = input("\nAutomatically optimize hyperparameters? (y/n): ").lower() == 'y'
    
    # Alpha parameter for regularized regression
    alpha = 1.0
    l1_ratio = 0.5
    
    if not auto_optimize and variant != 'standard':
        try:
            alpha = float(input("Enter alpha (regularization strength, default=1.0): ") or 1.0)
            if variant == 'elasticnet':
                l1_ratio = float(input("Enter l1_ratio (0-1, default=0.5): ") or 0.5)
                while not (0 <= l1_ratio <= 1):
                    print("l1_ratio must be between 0 and 1.")
                    l1_ratio = float(input("Enter l1_ratio (0-1, default=0.5): ") or 0.5)
        except ValueError:
            print("Invalid input. Using default values.")
    
    # Train the model
    print(f"\nTraining {variant} regression model...")
    print(f"Target: {target_column}")
    print(f"Features: {', '.join(feature_columns)}")
    
    model, metrics, X_train, X_test, y_train, y_test = train_linear_regression_model(
        data=data,
        target_column=target_column,
        feature_columns=feature_columns,
        variant=variant,
        alpha=alpha,
        l1_ratio=l1_ratio,
        auto_optimize=auto_optimize
    )
    
    # Display model performance
    print("\nModel Performance:")
    print(f"R-squared: {metrics['r2']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    # Display coefficients for interpretability
    if variant == 'standard':
        print("\nCoefficients:")
        # For one-hot encoded categorical features, we need to map coefficients correctly
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            regressor = model.named_steps['regressor']
            
            # Get feature names after preprocessing (including one-hot encoded categories)
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
                coefficients = regressor.coef_
                
                if len(feature_names) == len(coefficients):
                    for name, coef in zip(feature_names, coefficients):
                        print(f"{name}: {coef:.4f}")
                else:
                    # If there's a mismatch (possible with older sklearn versions)
                    print("Cannot match coefficients to feature names")
                    for i, coef in enumerate(regressor.coef_):
                        print(f"Feature {i+1}: {coef:.4f}")
            else:
                # Fallback for older sklearn versions
                for i, coef in enumerate(regressor.coef_):
                    print(f"Feature {i+1}: {coef:.4f}")
                
        print(f"Intercept: {model.intercept_:.4f}")
    
    return model, metrics, X_train, X_test, y_train, y_test
