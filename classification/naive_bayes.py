import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_naive_bayes_model(data, target_column, feature_columns, variant='gaussian', 
                          auto_optimize=False, test_size=0.25, random_state=None):
    """
    Train a Naive Bayes model on the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for training
    target_column : str
        The name of the target column
    feature_columns : list
        List of feature column names
    variant : str, default='gaussian'
        Type of Naive Bayes classifier: 'gaussian', 'multinomial', or 'bernoulli'
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
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle target if it's categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Select the appropriate NB variant
    if variant == 'gaussian':
        nb_classifier = GaussianNB()
    elif variant == 'multinomial':
        nb_classifier = MultinomialNB()
    elif variant == 'bernoulli':
        nb_classifier = BernoulliNB()
    else:
        raise ValueError(f"Unknown Naive Bayes variant: {variant}. Choose from 'gaussian', 'multinomial', or 'bernoulli'.")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', nb_classifier)
    ])
    
    # Auto-optimize if requested
    if auto_optimize:
        param_grid = {}
        
        if variant == 'gaussian':
            param_grid = {
                'classifier__var_smoothing': np.logspace(0, -9, 10)
            }
        elif variant == 'multinomial':
            param_grid = {
                'classifier__alpha': np.logspace(-3, 3, 7)
            }
        elif variant == 'bernoulli':
            param_grid = {
                'classifier__alpha': np.logspace(-3, 3, 7),
                'classifier__binarize': [0.0, 0.25, 0.5, 0.75]
            }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        pipeline = grid_search.best_estimator_
    else:
        # Train with default parameters
        pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
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
    pipeline.label_encoder = target_encoder
    pipeline.feature_columns = feature_columns
    pipeline.target_column = target_column
    
    # Return model, metrics, and data
    return pipeline, metrics, X_train, X_test, y_train, y_test
