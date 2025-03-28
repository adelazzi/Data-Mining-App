import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_decision_tree_model(data, target_column, feature_columns, max_depth=None, 
                             min_samples_split=2, criterion='gini', auto_optimize=False,
                             test_size=0.25, random_state=None):
    """
    Train a Decision Tree model on the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for training
    target_column : str
        The name of the target column
    feature_columns : list
        List of feature column names
    max_depth : int, default=None
        The maximum depth of the tree
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node
    criterion : str, default='gini'
        The function to measure the quality of a split: 'gini' or 'entropy'
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
    
    # Handle target if it's categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Create preprocessing for categorical features
    preprocessor = None
    if categorical_cols:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough',  # Keep numeric columns as they are
            verbose_feature_names_out=False  # Simplify output feature names
        )
    
    # Create pipeline
    pipeline_steps = []
    if preprocessor:
        pipeline_steps.append(('preprocessor', preprocessor))
    
    # Add the Decision Tree model
    if auto_optimize:
        # Use GridSearchCV to find optimal parameters
        pipeline_steps.append(('tree', DecisionTreeClassifier()))
        pipeline = Pipeline(pipeline_steps)
        
        param_grid = {
            'tree__max_depth': [None, 5, 10, 15, 20, 25],
            'tree__min_samples_split': [2, 5, 10],
            'tree__criterion': ['gini', 'entropy'],
            'tree__min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        pipeline = grid_search.best_estimator_
    else:
        # Use provided parameters
        pipeline_steps.append(('tree', DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state
        )))
        pipeline = Pipeline(pipeline_steps)
        
        # Train the model
        pipeline.fit(X_train, y_train)
    
    # If preprocessor exists, store the feature names after transformation
    if preprocessor:
        # Set feature names for correct prediction later
        pipeline.feature_names_in_ = feature_columns
        
        # If the pipeline has a preprocessor, get the feature names out
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            pipeline.feature_names_out_ = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
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
    
    # Store encoders and metadata in the model for later use
    pipeline.label_encoder = target_encoder
    pipeline.feature_columns = feature_columns
    pipeline.target_column = target_column
    
    # Store the tree for visualization separately instead of trying to add a property
    # Get the actual tree from the pipeline for visualization
    if 'tree' in pipeline.named_steps:
        pipeline.tree_model = pipeline.named_steps['tree']
        # Don't directly set the classes_ property
        pipeline.tree_classes = pipeline.named_steps['tree'].classes_
    else:
        # If no tree step found, handle gracefully
        pipeline.tree_model = None
        pipeline.tree_classes = None
    
    # Override the predict method to handle feature names
    original_predict = pipeline.predict
    def predict_wrapper(X):
        # If X is a DataFrame, extract only the required columns in the correct order
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in feature_columns):
                X = X[feature_columns]
            # If feature_columns not in X but X has the right number of columns, just use X
        
        # If X is a numpy array or list without column names, keep as is
        return original_predict(X)
    
    # Replace the predict method
    pipeline.predict = predict_wrapper
    
    # Return model, metrics, and data
    return pipeline, metrics, X_train, X_test, y_train, y_test
