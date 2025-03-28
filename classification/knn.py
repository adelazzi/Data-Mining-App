import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline

def train_knn_model(data, target_column, feature_columns, n_neighbors=5, weights='uniform',
                   metric='minkowski', auto_optimize=False, test_size=0.25, random_state=None):
    """
    Train a KNN model on the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for training
    target_column : str
        The name of the target column
    feature_columns : list
        List of feature column names
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='uniform'
        Weight function used in prediction: 'uniform' or 'distance'
    metric : str, default='minkowski'
        Distance metric for the tree
    auto_optimize : bool, default=False
        Whether to use GridSearchCV to find optimal parameters
    test_size : float, default=0.25
        The proportion of the dataset to include in the test split
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split
        
    Returns:
    --------
    tuple
        (model, metrics, X_train, X_test, y_train, y_test)
    """
    # Extract features and target
    X = data[feature_columns].copy()
    y = data[target_column].copy()
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        encoders[col] = encoder
    
    # Handle target if it's categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Create pipeline
    pipeline_steps = []
    
    # Add scaling only for numeric features
    if numeric_cols:
        pipeline_steps.append(('scaler', StandardScaler()))
        
    # Add the KNN model
    if auto_optimize:
        # Use GridSearchCV to find optimal parameters
        pipeline_steps.append(('knn', KNeighborsClassifier()))
        pipeline = Pipeline(pipeline_steps)
        
        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_params = grid_search.best_params_
        n_neighbors = best_params['knn__n_neighbors']
        weights = best_params['knn__weights']
        metric = best_params['knn__metric']
        
        # Train final model with best parameters
        model = Pipeline(pipeline_steps[:-1] + [('knn', KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            metric=metric
        ))])
    else:
        # Use provided parameters
        pipeline_steps.append(('knn', KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            metric=metric
        )))
        model = Pipeline(pipeline_steps)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }
    
    # Store encoders in the model for later use
    model.feature_encoders = encoders
    model.label_encoder = target_encoder
    model.feature_columns = feature_columns
    model.target_column = target_column
    
    # Return model, metrics, and data
    return model, metrics, X_train, X_test, y_train, y_test
