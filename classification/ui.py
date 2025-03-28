import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.tree import plot_tree

from .knn import train_knn_model
from .naive_bayes import train_naive_bayes_model
from .decision_tree import train_decision_tree_model
from .linear_regression import train_linear_regression_model
from .neural_network import train_neural_network_model

def show_classification_dialog(parent, data, model_type, existing_model=None, target_column=None, feature_columns=None):
    """
    Show a dialog for model settings and training
    
    Parameters:
    - parent: Parent window
    - data: DataFrame containing the data
    - model_type: Type of model ('knn', 'naive_bayes', 'decision_tree', 'linear_regression', 'neural_network')
    - existing_model: Optional, existing model for evaluation
    - target_column: Optional, target column name
    - feature_columns: Optional, list of feature column names
    
    Returns:
    - model: Trained model
    - model_type: Type of model
    - target_column: Target column name
    - feature_columns: List of feature column names
    - metrics: Dictionary of evaluation metrics
    """
    dialog = tk.Toplevel(parent)
    dialog.title(f"{'Evaluate' if existing_model else 'Train'} {model_type.replace('_', ' ').title()} Model")
    dialog.geometry("800x600")
    dialog.grab_set()  # Make the dialog modal
    
    # Container for the return value
    return_data = [None]
    
    # Create notebook for tab interfaces - simplified styling
    notebook = ttk.Notebook(dialog)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create tabs with basic styling
    tab_settings = ttk.Frame(notebook)
    tab_results = ttk.Frame(notebook)
    tab_visualization = ttk.Frame(notebook)
    
    notebook.add(tab_settings, text="Settings")
    notebook.add(tab_results, text="Results")
    
    # Add the appropriate visualization tab based on model type
    if model_type in ['knn', 'naive_bayes', 'decision_tree', 'neural_network']:
        notebook.add(tab_visualization, text="Visualization")
    elif model_type in ['linear_regression']:
        notebook.add(tab_visualization, text="Residual Plot")
    
    # Settings tab - simplified layout
    settings_frame = ttk.LabelFrame(tab_settings, text="Model Settings")
    settings_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Target column selection
    ttk.Label(settings_frame, text="Target Column:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
    
    target_var = tk.StringVar()
    target_dropdown = ttk.Combobox(settings_frame, textvariable=target_var, values=list(data.columns), width=30)
    target_dropdown.grid(row=0, column=1, sticky='w', padx=5, pady=5)
    
    # If target column is provided, select it
    if target_column:
        target_var.set(target_column)
    
    # Feature columns selection frame - basic styling
    features_frame = ttk.LabelFrame(tab_settings, text="Feature Selection")
    features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    ttk.Label(features_frame, text="Select features to use:").pack(anchor='w', padx=5, pady=5)
    
    # Create a "Select All" checkbox
    select_all_var = tk.BooleanVar()
    def toggle_select_all():
        if select_all_var.get():
            features_listbox.select_set(0, tk.END)  # Select all items
        else:
            features_listbox.select_clear(0, tk.END)  # Deselect all items

    select_all_checkbox = ttk.Checkbutton(
        features_frame, text="Select All", variable=select_all_var, command=toggle_select_all
    )
    select_all_checkbox.pack(anchor='w', padx=5, pady=5)
    
    # Create a listbox for feature selection - basic styling
    features_listbox = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, height=10)
    features_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Add scrollbar to listbox
    listbox_scrollbar = ttk.Scrollbar(features_listbox, orient="vertical", command=features_listbox.yview)
    listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    features_listbox.config(yscrollcommand=listbox_scrollbar.set)
    
    # Add columns to listbox
    for col in data.columns:
        features_listbox.insert(tk.END, col)
    
    # If feature columns are provided, select them
    if feature_columns:
        for i, col in enumerate(data.columns):
            if col in feature_columns:
                features_listbox.selection_set(i)
    
    # Test size - basic layout
    test_size_frame = ttk.Frame(settings_frame)
    test_size_frame.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
    
    ttk.Label(test_size_frame, text="Test Size:").pack(side=tk.LEFT, padx=5)
    test_size_var = tk.StringVar(value="0.3")
    test_size_entry = ttk.Entry(test_size_frame, textvariable=test_size_var, width=5)
    test_size_entry.pack(side=tk.LEFT, padx=5)
    
    # Random state - basic layout
    random_state_frame = ttk.Frame(settings_frame)
    random_state_frame.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)
    
    ttk.Label(random_state_frame, text="Random State:").pack(side=tk.LEFT, padx=5)
    random_state_var = tk.StringVar(value="42")
    random_state_entry = ttk.Entry(random_state_frame, textvariable=random_state_var, width=5)
    random_state_entry.pack(side=tk.LEFT, padx=5)
    
    # Add model-specific settings with simpler layout
    if model_type == 'knn':
        # n_neighbors for KNN
        knn_frame = ttk.Frame(settings_frame)
        knn_frame.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(knn_frame, text="Number of Neighbors (k):").pack(side=tk.LEFT, padx=5)
        k_var = tk.StringVar(value="5")
        k_entry = ttk.Entry(knn_frame, textvariable=k_var, width=5)
        k_entry.pack(side=tk.LEFT, padx=5)
    
    elif model_type == 'decision_tree':
        # Max depth for Decision Tree
        dt_frame1 = ttk.Frame(settings_frame)
        dt_frame1.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(dt_frame1, text="Maximum Tree Depth:").pack(side=tk.LEFT, padx=5)
        max_depth_var = tk.StringVar(value="")
        max_depth_entry = ttk.Entry(dt_frame1, textvariable=max_depth_var, width=5)
        max_depth_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(dt_frame1, text="(empty for unlimited)").pack(side=tk.LEFT, padx=5)
        
        # Min samples split for Decision Tree
        dt_frame2 = ttk.Frame(settings_frame)
        dt_frame2.grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(dt_frame2, text="Minimum Samples to Split:").pack(side=tk.LEFT, padx=5)
        min_samples_var = tk.StringVar(value="2")
        min_samples_entry = ttk.Entry(dt_frame2, textvariable=min_samples_var, width=5)
        min_samples_entry.pack(side=tk.LEFT, padx=5)
    
    elif model_type == 'neural_network':
        # Hidden layers for Neural Network
        nn_frame1 = ttk.Frame(settings_frame)
        nn_frame1.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(nn_frame1, text="Hidden Layer Sizes:").pack(side=tk.LEFT, padx=5)
        hidden_layers_var = tk.StringVar(value="100,100")
        hidden_layers_entry = ttk.Entry(nn_frame1, textvariable=hidden_layers_var, width=15)
        hidden_layers_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(nn_frame1, text="(comma-separated)").pack(side=tk.LEFT, padx=5)
        
        # Learning rate for Neural Network
        nn_frame2 = ttk.Frame(settings_frame)
        nn_frame2.grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(nn_frame2, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
        learning_rate_var = tk.StringVar(value="0.001")
        learning_rate_entry = ttk.Entry(nn_frame2, textvariable=learning_rate_var, width=8)
        learning_rate_entry.pack(side=tk.LEFT, padx=5)
        
        # Max iterations for Neural Network
        nn_frame3 = ttk.Frame(settings_frame)
        nn_frame3.grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(nn_frame3, text="Max Iterations:").pack(side=tk.LEFT, padx=5)
        max_iter_var = tk.StringVar(value="200")
        max_iter_entry = ttk.Entry(nn_frame3, textvariable=max_iter_var, width=8)
        max_iter_entry.pack(side=tk.LEFT, padx=5)
    
    # Results display - simple text widget with scrollbar
    results_text_frame = ttk.Frame(tab_results)
    results_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    results_text = tk.Text(results_text_frame, wrap=tk.WORD, height=15)
    results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    results_scrollbar = ttk.Scrollbar(results_text_frame, orient="vertical", command=results_text.yview)
    results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    results_text.config(yscrollcommand=results_scrollbar.set)
    results_text.config(state=tk.DISABLED)
    
    # Plot frame for visualization - basic container
    visualization_frame = ttk.Frame(tab_visualization)
    visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Function to train the model
    def train_model():
        try:
            # Get selected target column
            selected_target = target_var.get()
            if not selected_target:
                messagebox.showerror("Error", "Please select a target column")
                return
            
            # Get selected feature columns
            selected_indices = features_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select at least one feature column")
                return
            
            selected_features = [features_listbox.get(i) for i in selected_indices]
            
            # Check that the target is not also selected as a feature
            if selected_target in selected_features:
                messagebox.showerror("Error", "Target column cannot also be a feature")
                return
            
            # Get test size and random state
            try:
                test_size = float(test_size_var.get())
                if not (0 < test_size < 1):
                    messagebox.showerror("Error", "Test size must be between 0 and 1")
                    return
            except ValueError:
                messagebox.showerror("Error", "Test size must be a number")
                return
            
            try:
                random_state = int(random_state_var.get())
            except ValueError:
                messagebox.showerror("Error", "Random state must be an integer")
                return
            
            # Train the model based on type
            if model_type == 'knn':
                try:
                    k = int(k_var.get())
                    if k <= 0:
                        messagebox.showerror("Error", "Number of neighbors must be positive")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Number of neighbors must be an integer")
                    return
                
                model, metrics, X_train, X_test, y_train, y_test = train_knn_model(
                    data, 
                    selected_target, 
                    selected_features, 
                    n_neighbors=k, 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Update results for classification
                update_classification_results(model, metrics, selected_target, selected_features, X_train, X_test)
                
            elif model_type == 'naive_bayes':
                model, metrics, X_train, X_test, y_train, y_test = train_naive_bayes_model(
                    data, 
                    selected_target, 
                    selected_features, 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Update results for classification
                update_classification_results(model, metrics, selected_target, selected_features, X_train, X_test)
                
            elif model_type == 'decision_tree':
                # Get Decision Tree specific params
                try:
                    max_depth = None if max_depth_var.get() == "" else int(max_depth_var.get())
                    if max_depth is not None and max_depth <= 0:
                        messagebox.showerror("Error", "Max depth must be positive")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Max depth must be an integer")
                    return
                
                try:
                    min_samples_split = int(min_samples_var.get())
                    if min_samples_split < 2:
                        messagebox.showerror("Error", "Min samples split must be at least 2")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Min samples split must be an integer")
                    return
                
                model, metrics, X_train, X_test, y_train, y_test = train_decision_tree_model(
                    data, 
                    selected_target, 
                    selected_features, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Update results for classification
                update_classification_results(model, metrics, selected_target, selected_features, X_train, X_test)
                
                # Create decision tree visualization
                create_tree_visualization(model, selected_features)
                
            elif model_type == 'linear_regression':
                model, metrics, X_train, X_test, y_train, y_test = train_linear_regression_model(
                    data, 
                    selected_target, 
                    selected_features, 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Update results for regression
                update_regression_results(model, metrics, selected_target, selected_features, X_train, X_test, y_test, y_train)
                
            elif model_type == 'neural_network':
                # Get Neural Network specific params
                try:
                    hidden_layers = tuple(int(x.strip()) for x in hidden_layers_var.get().split(','))
                    if any(layers <= 0 for layers in hidden_layers):
                        messagebox.showerror("Error", "Hidden layer sizes must be positive")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Hidden layer sizes must be integers separated by commas")
                    return
                
                try:
                    learning_rate = float(learning_rate_var.get())
                    if learning_rate <= 0:
                        messagebox.showerror("Error", "Learning rate must be positive")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Learning rate must be a number")
                    return
                
                try:
                    max_iter = int(max_iter_var.get())
                    if max_iter <= 0:
                        messagebox.showerror("Error", "Max iterations must be positive")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Max iterations must be an integer")
                    return
                
                model, metrics, X_train, X_test, y_train, y_test = train_neural_network_model(
                    data, 
                    selected_target, 
                    selected_features, 
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Update results based on whether the NN is doing classification or regression
                if 'accuracy' in metrics:
                    # Classification metrics
                    update_classification_results(model, metrics, selected_target, selected_features, X_train, X_test)
                else:
                    # Regression metrics
                    update_regression_results(model, metrics, selected_target, selected_features, X_train, X_test, y_test, y_train)
            
            else:
                messagebox.showerror("Error", f"Unknown model type: {model_type}")
                return
            
            # Switch to results tab
            notebook.select(1)
            
            # Store the result
            return_data[0] = (model, model_type, selected_target, selected_features)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
    
    def update_classification_results(model, metrics, selected_target, selected_features, X_train, X_test):
        """Update the results display for classification models"""
        # Update results text
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        
        results_text.insert(tk.END, f"Model: {model_type.replace('_', ' ').title()}\n")
        results_text.insert(tk.END, f"Target Column: {selected_target}\n")
        results_text.insert(tk.END, f"Features: {', '.join(selected_features)}\n\n")
        
        results_text.insert(tk.END, f"Training Set Size: {len(X_train)}\n")
        results_text.insert(tk.END, f"Test Set Size: {len(X_test)}\n\n")
        
        results_text.insert(tk.END, f"Accuracy: {metrics['accuracy']:.4f}\n")
        results_text.insert(tk.END, f"Precision: {metrics['precision']:.4f}\n")
        results_text.insert(tk.END, f"Recall: {metrics['recall']:.4f}\n")
        results_text.insert(tk.END, f"F1 Score: {metrics['f1']:.4f}\n\n")
        
        # Add the full classification report
        results_text.insert(tk.END, "Classification Report:\n")
        results_text.insert(tk.END, f"{metrics['classification_report']}\n")
        
        results_text.config(state=tk.DISABLED)
        
        # Create confusion matrix visualization - simple style
        for widget in visualization_frame.winfo_children():
            widget.destroy()
        
        # Create a figure with one subplot for the confusion matrix
        fig = plt.figure(figsize=(10, 6))
        
        # Create confusion matrix plot
        ax1 = fig.add_subplot(111)
        cm = metrics['confusion_matrix']
        
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.figure.colorbar(im, ax=ax1)
        
        # Try to get original class names if label encoder was used
        if hasattr(model, 'label_encoder') and model.label_encoder is not None:
            classes = np.unique(model.label_encoder.classes_)
        else:
            classes = np.arange(len(cm))
            
        # Show all ticks
        ax1.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, 
               yticklabels=classes,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Embed the plot with a simple canvas
        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add save button
        save_button = ttk.Button(visualization_frame, text="Save Plot", 
                               command=lambda: fig.savefig(f"{model_type}_confusion_matrix.png"))
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def update_regression_results(model, metrics, selected_target, selected_features, X_train, X_test, y_test, y_train):
        """Update the results display for regression models"""
        # Update results text
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        
        results_text.insert(tk.END, f"Model: {model_type.replace('_', ' ').title()}\n")
        results_text.insert(tk.END, f"Target Column: {selected_target}\n")
        results_text.insert(tk.END, f"Features: {', '.join(selected_features)}\n\n")
        
        results_text.insert(tk.END, f"Training Set Size: {len(X_train)}\n")
        results_text.insert(tk.END, f"Test Set Size: {len(X_test)}\n\n")
        
        results_text.insert(tk.END, f"R² Score: {metrics['r2']:.4f}\n")
        results_text.insert(tk.END, f"Mean Absolute Error: {metrics['mae']:.4f}\n")
        results_text.insert(tk.END, f"Mean Squared Error: {metrics['mse']:.4f}\n")
        results_text.insert(tk.END, f"Root Mean Squared Error: {metrics['rmse']:.4f}\n\n")
        
        # If it's linear regression, show coefficients
        if model_type == 'linear_regression' and hasattr(model, 'coef_'):
            results_text.insert(tk.END, "Coefficients:\n")
            
            coef_array = model.coef_
            # Handle different shapes of coefficient arrays
            if coef_array.ndim == 1:
                coefs = coef_array
            else:
                coefs = coef_array[0]
                
            for i, feature in enumerate(selected_features):
                if i < len(coefs):
                    results_text.insert(tk.END, f"{feature}: {coefs[i]:.4f}\n")
            
            results_text.insert(tk.END, f"\nIntercept: {model.intercept_:.4f}\n")
        
        results_text.config(state=tk.DISABLED)
        
        # Create regression visualization - simple scatter plot
        for widget in visualization_frame.winfo_children():
            widget.destroy()
        
        # Predict on test data for visualization
        y_pred = model.predict(X_test)
        
        # Create residual plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot predicted vs actual
        ax.scatter(y_pred, y_test, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Predicted vs Actual Values')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        
        # Embed the plot
        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add save button
        save_button = ttk.Button(visualization_frame, text="Save Plot", 
                               command=lambda: fig.savefig(f"{model_type}_regression_plot.png"))
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def create_tree_visualization(model, feature_names):
        """Create a visualization of the decision tree"""
        # Create tree visualization for Decision Tree models
        for widget in visualization_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # If tree is too deep, limit the display
        max_depth_to_display = min(3, model.tree_.max_depth) if hasattr(model, 'tree_') else 3
        
        try:
            # Plot the tree with simpler styling
            plot_tree(model.named_steps['tree'] if hasattr(model, 'named_steps') else model, 
                     filled=True, 
                     feature_names=feature_names,
                     max_depth=max_depth_to_display, 
                     ax=ax,
                     fontsize=10)
        except Exception as e:
            # If tree visualization fails, show error message
            ax.text(0.5, 0.5, f"Tree visualization failed:\n{str(e)}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # Add title
        ax.set_title(f"Decision Tree (limited to depth {max_depth_to_display})")
        
        fig.tight_layout()
        
        # Embed the plot
        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add save button
        save_button = ttk.Button(visualization_frame, text="Save Tree", 
                               command=lambda: fig.savefig("decision_tree.png"))
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    # Buttons frame - basic layout
    buttons_frame = ttk.Frame(dialog)
    buttons_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Train/Evaluate button
    train_button = ttk.Button(
        buttons_frame, 
        text="Evaluate" if existing_model else "Train", 
        command=train_model
    )
    train_button.pack(side=tk.RIGHT, padx=5)
    
    # Cancel button
    cancel_button = ttk.Button(buttons_frame, text="Close", command=dialog.destroy)
    cancel_button.pack(side=tk.RIGHT, padx=5)
    
    # If existing model is provided, automatically train/evaluate
    if existing_model:
        # We need to wait a bit for the UI to initialize
        dialog.after(100, train_model)
    
    # Wait for the dialog to close
    dialog.wait_window()
    
    # Return the selected options and model
    return return_data[0]

def predict_new_value(parent, model, target_column, feature_columns, model_type):
    """
    Dialog to predict the class or value of a new input
    
    Parameters:
    - parent: Parent window
    - model: Trained model
    - target_column: Target column name
    - feature_columns: List of feature column names
    - model_type: Type of model
    """
    dialog = tk.Toplevel(parent)
    dialog.title(f"Predict using {model_type.replace('_', ' ').title()} Model")
    dialog.geometry("600x500")
    dialog.grab_set()  # Make the dialog modal
    
    # Style configuration
    style = ttk.Style()
    style.configure("Title.TLabel", font=('Helvetica', 12, 'bold'))
    style.configure("Result.TLabel", font=('Helvetica', 11), foreground="#0066cc")
    style.configure("Accent.TButton", font=('Helvetica', 10), background="#4CAF50")
    
    # Main container with padding
    main_container = ttk.Frame(dialog, padding="20 15 20 15")
    main_container.pack(fill=tk.BOTH, expand=True)
    
    # Header with icon and title
    header_frame = ttk.Frame(main_container)
    header_frame.pack(fill=tk.X, pady=(0, 15))
    
    # Title with model info
    ttk.Label(
        header_frame, 
        text=f"Predict {target_column} using {model_type.replace('_', ' ').title()}",
        style="Title.TLabel"
    ).pack(side=tk.LEFT, pady=10)
    
    # Separator after header
    ttk.Separator(main_container, orient='horizontal').pack(fill=tk.X, pady=5)
    
    # Frame for feature inputs with better styling and scrolling
    input_container = ttk.Frame(main_container)
    input_container.pack(fill=tk.BOTH, expand=True, pady=10)
    
    ttk.Label(input_container, text="Enter feature values:", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 10))
    
    # Add a canvas with scrollbar for many features
    canvas = tk.Canvas(input_container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(input_container, orient="vertical", command=canvas.yview)
    
    features_frame = ttk.Frame(canvas)
    
    # Configure canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Add features frame to canvas
    canvas_frame = canvas.create_window((0, 0), window=features_frame, anchor="nw")
    
    # Dictionary to store entry widgets and variables
    feature_entries = {}
    
    # Create entry fields for each feature with improved layout
    for i, feature in enumerate(feature_columns):
        frame = ttk.Frame(features_frame, padding="5 3 5 3")
        frame.pack(fill=tk.X, pady=5)
        
        # Label with fixed width for better alignment
        label = ttk.Label(frame, text=f"{feature}:", width=20, anchor='e')
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=var, width=30)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        feature_entries[feature] = var
    
    # Update canvas scrollregion when the size of the frame changes
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    features_frame.bind("<Configure>", configure_scroll_region)
    
    # Make sure the canvas expands to the width of the container
    def configure_canvas(event):
        canvas.itemconfig(canvas_frame, width=event.width)
    
    canvas.bind("<Configure>", configure_canvas)
    
    # Function to generate random values for all fields
    def fill_random_values():
        import random
        
        for feature in feature_columns:
            # Generate a reasonable random value based on feature name
            if any(keyword in feature.lower() for keyword in ['age', 'year', 'count', 'num']):
                # Integer values for features that sound like counts or years
                value = str(random.randint(1, 100))
            elif any(keyword in feature.lower() for keyword in ['price', 'cost', 'income', 'rate']):
                # Decimal values for money-related features
                value = f"{random.uniform(10, 1000):.2f}"
            elif any(keyword in feature.lower() for keyword in ['ratio', 'percent', 'proportion']):
                # Values between 0 and 1 for ratio-like features
                value = f"{random.random():.3f}"
            else:
                # Default to a floating point number between 0 and 10
                value = f"{random.uniform(0, 10):.2f}"
                
            feature_entries[feature].set(value)
    
    # Separator before results section
    ttk.Separator(main_container, orient='horizontal').pack(fill=tk.X, pady=10)
    
    # Results frame with improved styling
    results_frame = ttk.LabelFrame(main_container, text="Prediction Result", padding="10 5 10 10")
    results_frame.pack(fill=tk.X, pady=10)
    
    # Icon indicator (empty initially)
    result_container = ttk.Frame(results_frame)
    result_container.pack(fill=tk.X, expand=True, pady=5)
    
    indicator_label = ttk.Label(result_container, text="•", font=('Helvetica', 16), foreground="gray")
    indicator_label.pack(side=tk.LEFT, padx=(0, 10))
    
    prediction_label = ttk.Label(
        result_container, 
        text="Enter values and click Predict",
        style="Result.TLabel"
    )
    prediction_label.pack(side=tk.LEFT, pady=5)
    
    # Function to make prediction with visual feedback
    def make_prediction():
        try:
            # Show "calculating" state
            indicator_label.config(text="•", foreground="#FFA500")  # Orange dot
            prediction_label.config(text="Calculating prediction...")
            dialog.update()  # Force UI update
            
            # Collect input values
            input_values = []
            missing_fields = []
            
            for feature in feature_columns:
                value = feature_entries[feature].get().strip()
                
                # Check for empty fields
                if not value:
                    missing_fields.append(feature)
                    continue
                
                # Try to convert to a number if appropriate
                try:
                    value = float(value)
                    # Convert to int if it's a whole number
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    # Keep as string if not a number
                    pass
                
                input_values.append(value)
            
            # If missing fields, alert the user
            if missing_fields:
                indicator_label.config(text="✗", foreground="#FF0000")  # Red X
                prediction_label.config(text=f"Missing values for: {', '.join(missing_fields[:3])}" + 
                                            ("..." if len(missing_fields) > 3 else ""))
                return
            
            # Convert to numpy array
            input_array = np.array([input_values])
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            
            # Check if regression or classification
            is_regression = model_type in ['linear_regression'] or (
                model_type == 'neural_network' and isinstance(prediction, (int, float, np.number))
            )
            
            # Update UI with result
            indicator_label.config(text="✓", foreground="#00AA00")  # Green checkmark
            
            if is_regression:
                # For regression, display the predicted value
                prediction_label.config(text=f"Predicted {target_column}: {prediction:.4f}")
            else:
                # For classification, display the predicted class
                prediction_label.config(text=f"Predicted {target_column}: {prediction}")
            
        except Exception as e:
            indicator_label.config(text="✗", foreground="#FF0000")  # Red X
            prediction_label.config(text=f"Error: {str(e)}")
    
    # Buttons frame with improved styling
    buttons_frame = ttk.Frame(main_container)
    buttons_frame.pack(fill=tk.X, pady=(15, 0))
    
    # Help button (left side)
    help_button = ttk.Button(
        buttons_frame, 
        text="ℹ️ Help",
        command=lambda: messagebox.showinfo(
            "Help", 
            "Enter values for each feature and click 'Predict' to get a prediction.\n\n"
            f"This model predicts '{target_column}' based on {len(feature_columns)} features."
        )
    )
    help_button.pack(side=tk.LEFT, padx=5)
    
    # Random values button (left side)
    random_button = ttk.Button(
        buttons_frame, 
        text="Random Values",
        command=fill_random_values
    )
    random_button.pack(side=tk.LEFT, padx=5)
    
    # Clear button (left side)
    clear_button = ttk.Button(
        buttons_frame, 
        text="Clear All",
        command=lambda: [var.set("") for var in feature_entries.values()]
    )
    clear_button.pack(side=tk.LEFT, padx=5)
    
    # Predict button (right side with accent styling)
    predict_button = ttk.Button(
        buttons_frame, 
        text="Predict", 
        style="Accent.TButton",
        command=make_prediction
    )
    predict_button.pack(side=tk.RIGHT, padx=5)
    
    # Close button (right side)
    close_button = ttk.Button(
        buttons_frame, 
        text="Close", 
        command=dialog.destroy
    )
    close_button.pack(side=tk.RIGHT, padx=5)
    
    # Center dialog on parent
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    dialog.geometry(f"{width}x{height}+{x}+{y}")
