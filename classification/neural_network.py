import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, mean_squared_error, mean_absolute_error, 
                            r2_score, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_neural_network_model(data, target_column, feature_columns, hidden_layers=(100,),
                              learning_rate=0.001, max_iter=200, activation='relu',
                              task='auto', auto_optimize=False, test_size=0.25, random_state=None):
    """
    Train a Neural Network model on the dataset for either classification or regression.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for training
    target_column : str
        The name of the target column
    feature_columns : list
        List of feature column names
    hidden_layers : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer
    learning_rate : float, default=0.001
        Learning rate for weight updates
    max_iter : int, default=200
        Maximum number of iterations
    activation : str, default='relu'
        Activation function: 'identity', 'logistic', 'tanh', or 'relu'
    task : str, default='auto'
        The task type: 'auto' (detect automatically), 'classification', or 'regression'
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
    
    # Check if this is a classification or regression task
    is_classification = True
    if task == 'auto':
        # For numeric targets, check the number of unique values
        if pd.api.types.is_numeric_dtype(y):
            # If number of unique values is small compared to total data size, 
            # it's likely a classification problem
            unique_ratio = len(y.unique()) / len(y)
            is_classification = unique_ratio < 0.05  # arbitrary threshold
        else:
            # For categorical targets, it's classification
            is_classification = True
    elif task == 'classification':
        is_classification = True
    elif task == 'regression':
        is_classification = False
    else:
        raise ValueError(f"Unknown task: {task}. Choose from 'auto', 'classification', or 'regression'.")
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle target if it's categorical (for classification)
    target_encoder = None
    original_classes = None
    if is_classification and (y.dtype == 'object' or y.dtype.name == 'category'):
        target_encoder = LabelEncoder()
        original_classes = np.array(y.unique())  # Store original class labels
        y = target_encoder.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if is_classification and len(np.unique(y)) > 1 else None
    )
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Choose neural network model based on task
    if is_classification:
        nn_model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            activation=activation,
            random_state=random_state
        )
    else:
        nn_model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            activation=activation,
            random_state=random_state
        )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('nn', nn_model)
    ])
    
    # Auto-optimize if requested
    if auto_optimize:
        param_grid = {
            'nn__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'nn__activation': ['relu', 'tanh'],
            'nn__learning_rate_init': [0.001, 0.01, 0.1],
            'nn__alpha': [0.0001, 0.001, 0.01]
        }
        
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        pipeline = grid_search.best_estimator_
    else:
        # Train with provided parameters
        pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics based on task
    if is_classification:
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    # Store information in the model for later use
    pipeline.label_encoder = target_encoder
    pipeline.feature_columns = feature_columns
    pipeline.target_column = target_column
    pipeline.is_classification = is_classification
    pipeline.scaler = preprocessor
    
    # Store original class labels if this is a classification task
    if is_classification:
        pipeline.original_classes = original_classes
        # The actual classes_ is already available through the classifier in the pipeline
        # We don't need to set it manually, it can be accessed via:
        # pipeline.named_steps['nn'].classes_
    
    # Return model, metrics, and data
    return pipeline, metrics, X_train, X_test, y_train, y_test
