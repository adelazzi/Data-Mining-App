import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import math

# Import the reorganized functions from the classification package
from classification import (
    train_knn_model,
    train_naive_bayes_model,
    train_decision_tree_model,
    train_linear_regression_model,
    train_neural_network_model,
    show_classification_dialog,
    predict_new_value,
    export_classification_results_to_csv
)

# The original implementation of all methods has been moved to separate files
# in the classification directory. This file now acts as a wrapper that imports
# those functions from the package.

def predict_new_value(root, model, target_column, feature_columns, model_type, dataset=None):
    """
    Create a dialog to input values for prediction and show the result.
    Adds range info buttons to help users enter appropriate values.
    
    Args:
        root: The tkinter root window
        model: The trained model
        target_column: The name of the target column
        feature_columns: List of feature column names
        model_type: Type of the model (classification or regression)
        dataset: The dataset used for training (to get ranges)
    """
    # Create a new dialog window
    dialog = tk.Toplevel(root)
    dialog.title(f"Predict New Value - {model_type.replace('_', ' ').title()}")
    dialog.geometry("600x700")
    dialog.grab_set()  # Make window modal
    
    # Create main frame with scrollbar support
    main_frame = ttk.Frame(dialog, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Add a title
    title_label = ttk.Label(
        main_frame, 
        text=f"Enter Values for Prediction",
        font=("Segoe UI", 14, "bold")
    )
    title_label.pack(pady=(0, 20))
    
    # Create a canvas with scrollbar
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Dictionary to store entry widgets
    entries = {}
    entry_validation = {}
    
    # Calculate field ranges if dataset is provided
    field_ranges = {}
    field_types = {}
    categorical_values = {}
    
    if dataset is not None:
        for feature in feature_columns:
            if feature in dataset.columns:
                # Determine data type
                if pd.api.types.is_numeric_dtype(dataset[feature]):
                    field_types[feature] = 'numeric'
                    min_val = dataset[feature].min()
                    max_val = dataset[feature].max()
                    mean_val = dataset[feature].mean()
                    field_ranges[feature] = (min_val, max_val, mean_val)
                else:
                    field_types[feature] = 'categorical'
                    categorical_values[feature] = dataset[feature].unique().tolist()
    
    # Create input fields for each feature
    for i, feature in enumerate(feature_columns):
        # Create a frame for each feature row
        feature_frame = ttk.Frame(scrollable_frame)
        feature_frame.pack(fill="x", pady=8)
        
        # Label
        label = ttk.Label(feature_frame, text=f"{feature}:", width=20, anchor="e")
        label.pack(side="left", padx=(0, 10))
        
        # Entry or Combobox based on data type
        if feature in field_types and field_types[feature] == 'categorical' and feature in categorical_values:
            entry = ttk.Combobox(feature_frame, width=25, values=categorical_values[feature])
            if len(categorical_values[feature]) > 0:
                entry.set(categorical_values[feature][0])  # Default to first value
        else:
            entry = ttk.Entry(feature_frame, width=25)
            # Add default value if ranges are available
            if feature in field_ranges:
                entry.insert(0, f"{field_ranges[feature][2]:.4f}")  # Insert mean value
        
        entry.pack(side="left")
        entries[feature] = entry
        
        # Validation label to show errors
        validation_label = ttk.Label(feature_frame, text="", foreground="red")
        validation_label.pack(side="left", padx=5)
        entry_validation[feature] = validation_label
        
        # Range button for numeric fields
        if feature in field_ranges:
            min_val, max_val, mean_val = field_ranges[feature]
            
            def show_range(feat=feature, min_v=min_val, max_v=max_val, mean_v=mean_val):
                messagebox.showinfo(
                    "Value Range", 
                    f"Range for {feat}:\nMinimum: {min_v:.4f}\nMaximum: {max_v:.4f}\nAverage: {mean_v:.4f}"
                )
            
            range_btn = ttk.Button(
                feature_frame, 
                text="?", 
                width=3,
                command=show_range
            )
            range_btn.pack(side="left", padx=5)
            
            # Add a slider for numeric fields
            if pd.api.types.is_numeric_dtype(dataset[feature]) and max_val - min_val <= 100:
                slider_frame = ttk.Frame(scrollable_frame)
                slider_frame.pack(fill="x", pady=(0, 10))
                
                slider = ttk.Scale(
                    slider_frame, 
                    from_=min_val, 
                    to=max_val, 
                    orient="horizontal",
                    length=300
                )
                slider.set(mean_val)  # Set to mean value
                slider.pack(padx=(30, 0))
                
                # Update entry when slider changes
                def update_entry(val, feat=feature, entry_widget=entry):
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, f"{float(val):.4f}")
                
                slider.config(command=update_entry)
    
    # Separator
    ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
    
    # Result frame
    result_frame = ttk.Frame(scrollable_frame)
    result_frame.pack(fill="x", pady=10)
    
    result_label = ttk.Label(
        result_frame, 
        text="Prediction will appear here",
        font=("Segoe UI", 12)
    )
    result_label.pack(pady=10)
    
    # Frame for visualization
    viz_frame = ttk.Frame(scrollable_frame)
    viz_frame.pack(fill="both", expand=True, pady=10)
    
    # Function to validate inputs
    def validate_inputs():
        is_valid = True
        for feature, entry_widget in entries.items():
            validation_label = entry_validation[feature]
            value = entry_widget.get().strip()
            
            # Check if empty
            if not value:
                validation_label.config(text="Required")
                is_valid = False
                continue
                
            # Validate numeric fields
            if feature in field_types and field_types[feature] == 'numeric':
                try:
                    float_val = float(value)
                    # Check range
                    if feature in field_ranges:
                        min_val, max_val, _ = field_ranges[feature]
                        if float_val < min_val or float_val > max_val:
                            validation_label.config(text=f"Outside range")
                            # Still valid but with warning
                    else:
                        validation_label.config(text="")
                except ValueError:
                    validation_label.config(text="Not a number")
                    is_valid = False
            else:
                validation_label.config(text="")
                
        return is_valid
    
    # Function to perform prediction
    def do_prediction():
        # Clear previous visualization
        for widget in viz_frame.winfo_children():
            widget.destroy()
            
        # Validate inputs first
        if not validate_inputs():
            messagebox.showwarning("Input Validation", "Please correct the input errors before prediction.")
            return
            
        try:
            # Collect values from entries
            input_data = {}
            for feature, entry_widget in entries.items():
                try:
                    value = entry_widget.get().strip()
                    # Convert to appropriate type
                    if feature in field_types and field_types[feature] == 'numeric':
                        value = float(value)
                    input_data[feature] = value
                except Exception as e:
                    messagebox.showerror("Input Error", f"Invalid input for {feature}: {str(e)}")
                    return
            
            # Convert to DataFrame with a single row
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            if model_type in ["knn", "naive_bayes", "decision_tree"]:
                prediction = model.predict(input_df)
                prediction_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        prediction_proba = model.predict_proba(input_df)[0]
                    except:
                        pass
                
                result_text = f"Predicted {target_column}: {prediction[0]}"
                result_label.config(text=result_text, foreground="green", font=("Segoe UI", 12, "bold"))
                
                # Visualize probabilities if available
                if prediction_proba is not None:
                    fig, ax = plt.figure(figsize=(5, 3)), plt.axes()
                    classes = model.classes_
                    ax.bar(classes, prediction_proba)
                    ax.set_title("Prediction Probabilities")
                    ax.set_ylabel("Probability")
                    ax.set_ylim(0, 1)
                    
                    # Add canvas to display the plot
                    canvas = FigureCanvasTkAgg(fig, master=viz_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:  # Regression models
                prediction = model.predict(input_df)
                result_text = f"Predicted {target_column}: {prediction[0]:.4f}"
                result_label.config(text=result_text, foreground="green", font=("Segoe UI", 12, "bold"))
                
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")
            result_label.config(text=f"Error: {str(e)}", foreground="red")
    
    # Button frame
    button_frame = ttk.Frame(scrollable_frame)
    button_frame.pack(fill="x", pady=20)
    
    # Reset button
    def reset_form():
        for feature, entry_widget in entries.items():
            entry_widget.delete(0, tk.END)
            if feature in field_ranges:
                # Reset to mean for numeric fields
                entry_widget.insert(0, f"{field_ranges[feature][2]:.4f}")
            # Reset validation labels
            entry_validation[feature].config(text="")
        
        # Clear result
        result_label.config(text="Prediction will appear here", foreground="black", font=("Segoe UI", 12))
        
        # Clear visualization
        for widget in viz_frame.winfo_children():
            widget.destroy()
    
    reset_button = ttk.Button(
        button_frame, 
        text="Reset",
        command=reset_form
    )
    reset_button.pack(side="left", padx=5)
    
    # Predict button
    predict_button = ttk.Button(
        button_frame, 
        text="Predict",
        command=do_prediction,
        style="Accent.TButton"
    )
    predict_button.pack(side="left", padx=5)
    
    # Close button
    close_button = ttk.Button(
        button_frame, 
        text="Close",
        command=dialog.destroy
    )
    close_button.pack(side="right", padx=5)
