import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

def round_floats(df, decimal_places=6):
    """
    Round all float columns in a dataframe to the specified number of decimal places
    
    Args:
        df: DataFrame to process
        decimal_places: Number of decimal places to round to (default: 6)
        
    Returns:
        DataFrame with float values rounded
    """
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = df[col].round(decimal_places)
    return df

def clean_data(data):
    """
    Clean the data by:
    - Removing duplicate rows
    - Removing rows with too many missing values
    - Converting data types when appropriate
    - Handling categorical data by standardizing string formats
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Remove rows where more than 50% of values are missing
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    
    # Try to convert object columns to numeric where possible
    for col in df.select_dtypes(include=['object']):
        try:
            # Check if the column contains only numeric values first
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            # If less than 10% of values are NaN after conversion, assume it's numeric
            if numeric_conversion.isna().mean() < 0.1:
                df[col] = numeric_conversion
            else:
                # Otherwise, clean up categorical data by standardizing string format
                if df[col].dtype == 'object':
                    # Convert to string, strip whitespace, and convert to lowercase
                    df[col] = df[col].astype(str).str.strip().str.lower()
                    
                    # Replace common variations and abbreviations (example for Yes/No)
                    yes_variants = ['yes', 'y', 'true', 't', '1']
                    no_variants = ['no', 'n', 'false', 'f', '0']
                    
                    # Check if the column might be a yes/no column
                    unique_vals = df[col].unique()
                    if len(unique_vals) <= 5 and all(val.lower() in yes_variants + no_variants for val in unique_vals if pd.notna(val) and val != ''):
                        df[col] = df[col].apply(lambda x: 'yes' if str(x).lower() in yes_variants else 
                                               ('no' if str(x).lower() in no_variants else x))
        except:
            pass  # Keep as is if conversion fails
    
    # Round float values to 6 decimal places
    df = round_floats(df)
    
    return df

def normalize_data(data, handle_categorical=True, parent_window=None):
    """
    Normalize numeric columns in the dataset and optionally handle categorical data
    
    Args:
        data: DataFrame to normalize
        handle_categorical: Boolean to determine if categorical columns should be encoded
        parent_window: Parent window for progress dialog
        
    Returns:
        Normalized DataFrame
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Show progress bar if parent window is provided
    progress_window = None
    if parent_window is not None:
        total_steps = len(numeric_cols) + (len(categorical_cols) if handle_categorical else 0)
        progress_window, progress_bar, progress_label = show_progress_bar(
            parent_window, 
            "Normalizing Data",
            max_value=total_steps or 1  # Ensure at least 1 step
        )
    
    step_count = 0
    
    try:
        # Normalize each numeric column
        if numeric_cols:
            scaler = preprocessing.StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            # Round the normalized values to 6 decimal places
            df = round_floats(df)
            
            step_count += len(numeric_cols)
            # Update progress
            if progress_window is not None:
                progress_label.config(text="Normalizing numeric columns...")
                progress_bar["value"] = step_count
                progress_window.update()
        
        # Handle categorical columns if requested
        if handle_categorical and categorical_cols:
            if progress_window is not None:
                progress_label.config(text="Analyzing categorical columns...")
                progress_window.update()
            
            for i, col in enumerate(categorical_cols):
                # Check the cardinality (number of unique values)
                n_unique = df[col].nunique()
                
                # Different encoding strategies based on cardinality
                if n_unique <= 2:  # Binary categorical
                    # Label encode binary columns (0/1)
                    le = preprocessing.LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    
                    if progress_window is not None:
                        progress_label.config(text=f"Binary encoding: {col}")
                        progress_window.update()
                
                elif n_unique <= 10 and n_unique / len(df) < 0.05:  # Low cardinality & not too sparse
                    # One-hot encode columns with few unique values
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                    
                    if progress_window is not None:
                        progress_label.config(text=f"One-hot encoding: {col}")
                        progress_window.update()
                
                else:  # High cardinality
                    # For high cardinality categorical features, use a more sophisticated approach
                    # Calculate the frequency of each category and convert to a numerical value
                    value_counts = df[col].value_counts(normalize=True)
                    df[f"{col}_freq"] = df[col].map(value_counts)
                    
                    # Optionally, we can create a "is_rare" flag for rare categories
                    rare_threshold = 0.01  # 1% frequency
                    df[f"{col}_is_rare"] = df[col].map(value_counts < rare_threshold).astype(int)
                    
                    # Drop the original column to avoid redundancy
                    df = df.drop(col, axis=1)
                    
                    if progress_window is not None:
                        progress_label.config(text=f"Frequency encoding: {col}")
                        progress_window.update()
                
                step_count += 1
                if progress_window is not None:
                    progress_bar["value"] = step_count
                    progress_window.update()
    
    finally:
        # Close progress window if it exists
        if progress_window is not None:
            progress_window.destroy()
    
    return df

def handle_missing_values(data):
    """
    Handle missing values using various strategies with improved handling for categorical data
    """
    # Open a dialog to ask user how to handle missing values
    result = tk.Toplevel()
    result.title("Handle Missing Values")
    result.geometry("500x400")
    result.grab_set()  # Make the dialog modal
    
    # Create frame for options
    options_frame = ttk.Frame(result)
    options_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Create summary of missing values
    missing_info = pd.DataFrame({
        'Column': data.columns,
        'Missing Values': data.isnull().sum().values,
        'Percentage': data.isnull().sum().values / len(data) * 100,
        'Type': [str(data[col].dtype) for col in data.columns]
    })
    
    # Show summary
    ttk.Label(options_frame, text="Missing Values Summary:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
    
    # Create a treeview to display missing values info
    tree_frame = ttk.Frame(options_frame)
    tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    tree = ttk.Treeview(tree_frame, columns=list(missing_info.columns), show="headings")
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Set column headings
    for col in missing_info.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    # Insert data rows
    for i, row in missing_info.iterrows():
        values = list(row)
        if values[1] > 0:  # Only show columns with missing values
            tree.insert("", tk.END, values=values)
    
    tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
    
    # Strategy selection with improved options for categorical data
    strategy_frame = ttk.Frame(result)
    strategy_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(strategy_frame, text="Choose strategy for handling missing values:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
    
    strategy = tk.StringVar()
    strategy.set("smart_fill")  # Set smart fill as default
    
    ttk.Radiobutton(strategy_frame, text="Drop rows with missing values", variable=strategy, value="drop").grid(row=1, column=0, sticky='w', padx=20, pady=2)
    ttk.Radiobutton(strategy_frame, text="Smart fill (best method for each column type)", variable=strategy, value="smart_fill").grid(row=2, column=0, sticky='w', padx=20, pady=2)
    ttk.Radiobutton(strategy_frame, text="Fill numeric with mean, categorical with mode", variable=strategy, value="fill_mean_mode").grid(row=3, column=0, sticky='w', padx=20, pady=2)
    ttk.Radiobutton(strategy_frame, text="Forward fill (use previous values)", variable=strategy, value="ffill").grid(row=4, column=0, sticky='w', padx=20, pady=2)
    ttk.Radiobutton(strategy_frame, text="Fill with zero/empty", variable=strategy, value="zero").grid(row=5, column=0, sticky='w', padx=20, pady=2)
    
    # Add option for handling high cardinality categorical columns
    categorical_strategy = tk.StringVar()
    categorical_strategy.set("mode")
    
    categorical_frame = ttk.Frame(result)
    categorical_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(categorical_frame, text="Strategy for categorical columns:", font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=5)
    
    ttk.Radiobutton(categorical_frame, text="Mode (most frequent value)", variable=categorical_strategy, value="mode").grid(row=1, column=0, sticky='w', padx=20, pady=2)
    ttk.Radiobutton(categorical_frame, text="New category ('Unknown')", variable=categorical_strategy, value="unknown").grid(row=2, column=0, sticky='w', padx=20, pady=2)
    ttk.Radiobutton(categorical_frame, text="Most similar row's value", variable=categorical_strategy, value="similar").grid(row=3, column=0, sticky='w', padx=20, pady=2)
    
    # Variables to return
    return_data = [None]
    
    # Apply function with improved handling for categorical data
    def on_apply():
        selected_strategy = strategy.get()
        selected_cat_strategy = categorical_strategy.get()
        df = data.copy()
        
        try:
            if selected_strategy == "drop":
                df = df.dropna()
                messagebox.showinfo("Result", f"Dropped {len(data) - len(df)} rows with missing values.")
            
            elif selected_strategy == "smart_fill":
                # Intelligently handle each column based on its type and missing data pattern
                for column in df.columns:
                    # Skip columns with no missing values
                    if df[column].isna().sum() == 0:
                        continue
                        
                    # For numeric columns
                    if df[column].dtype.kind in 'ifc':
                        # For columns with less than 5% missing, use mean
                        if df[column].isna().mean() < 0.05:
                            df.loc[:, column] = df[column].fillna(df[column].mean())
                        # For columns with skew, use median
                        elif abs(df[column].skew()) > 1:
                            df.loc[:, column] = df[column].fillna(df[column].median())
                        # Otherwise use mean
                        else:
                            df.loc[:, column] = df[column].fillna(df[column].mean())
                    
                    # For categorical columns
                    else:
                        if selected_cat_strategy == "mode":
                            # Fill with the most common value
                            most_common = df[column].mode()[0] if not df[column].mode().empty else ""
                            df.loc[:, column] = df[column].fillna(most_common)
                            
                        elif selected_cat_strategy == "unknown":
                            # Fill with a new category "Unknown"
                            df.loc[:, column] = df[column].fillna("Unknown")
                            
                        elif selected_cat_strategy == "similar":
                            # This is a more complex approach - find the most similar row
                            # Only implement if we have enough non-missing data
                            if df[column].isna().mean() < 0.3:
                                # For each row with a missing value in this column
                                for idx in df[df[column].isna()].index:
                                    # Get the row data without the missing column
                                    row_data = df.loc[idx].drop(column)
                                    
                                    # Get all rows that have a value for this column
                                    complete_rows = df[~df[column].isna()]
                                    
                                    if len(complete_rows) > 0:
                                        # Create a simple similarity measure based on matching values
                                        similarities = []
                                        
                                        for comp_idx, comp_row in complete_rows.iterrows():
                                            # Count matching values in other columns
                                            matching_count = 0
                                            total_compared = 0
                                            
                                            for col in df.columns:
                                                if col != column and not pd.isna(row_data[col]) and not pd.isna(comp_row[col]):
                                                    total_compared += 1
                                                    if row_data[col] == comp_row[col]:
                                                        matching_count += 1
                                            
                                            # Calculate similarity score (avoid division by zero)
                                            similarity = matching_count / total_compared if total_compared > 0 else 0
                                            similarities.append((comp_idx, similarity))
                                        
                                        # Sort by similarity (highest first)
                                        similarities.sort(key=lambda x: x[1], reverse=True)
                                        
                                        # Get the value from the most similar row
                                        if similarities and similarities[0][1] > 0:
                                            most_similar_idx = similarities[0][0]
                                            df.loc[idx, column] = df.loc[most_similar_idx, column]
                                        else:
                                            # Fallback to mode if no similar rows
                                            most_common = df[column].mode()[0] if not df[column].mode().empty else ""
                                            df.loc[idx, column] = most_common
                                    else:
                                        # Fallback if no complete rows
                                        df.loc[idx, column] = ""
                            else:
                                # Fall back to mode for columns with too many missing values
                                most_common = df[column].mode()[0] if not df[column].mode().empty else ""
                                df.loc[:, column] = df[column].fillna(most_common)
                
                # Round float values after filling
                df = round_floats(df)
                messagebox.showinfo("Result", "Applied smart filling strategy to missing values.")
            
            elif selected_strategy == "fill_mean_mode":
                for column in df.columns:
                    if df[column].dtype.kind in 'ifc':  # if column is numeric
                        df.loc[:, column] = df[column].fillna(df[column].mean())
                    else:  # for categorical columns
                        df.loc[:, column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else "")
                # Round float values after filling
                df = round_floats(df)
                messagebox.showinfo("Result", "Filled missing values with mean/mode.")
            
            elif selected_strategy == "ffill":
                df = df.fillna(method='ffill')
                # For any remaining NaN values (at the beginning), use backfill
                df = df.fillna(method='bfill')
                # Round float values after filling
                df = round_floats(df)
                messagebox.showinfo("Result", "Filled missing values using forward/backward fill.")
            
            elif selected_strategy == "zero":
                for column in df.columns:
                    if df[column].dtype.kind in 'ifc':  # if column is numeric
                        df[column].fillna(0, inplace=True)
                    else:  # for categorical columns
                        df[column].fillna("", inplace=True)
                # Round float values after filling
                df = round_floats(df)
                messagebox.showinfo("Result", "Filled numeric missing values with 0, categorical with empty string.")
            
            return_data[0] = df
            result.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying strategy: {str(e)}")
    
    # Cancel function
    def on_cancel():
        result.destroy()
    
    # Buttons frame
    button_frame = ttk.Frame(result)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Button(button_frame, text="Apply", command=on_apply).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
    
    result.wait_window()
    return return_data[0] if return_data[0] is not None else data

def show_preprocessing_dialog(parent, data):
    """
    Show a dialog with advanced preprocessing options
    """
    dialog = tk.Toplevel(parent)
    dialog.title("Advanced Preprocessing")
    dialog.geometry("700x500")
    dialog.grab_set()  # Make the dialog modal
    
    # Container for the return value
    return_data = [None]
    
    # Create notebook for tab interfaces
    notebook = ttk.Notebook(dialog)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create tabs
    tab_normalize = ttk.Frame(notebook)
    tab_scaling = ttk.Frame(notebook)
    tab_transform = ttk.Frame(notebook)
    tab_encoding = ttk.Frame(notebook)
    
    notebook.add(tab_normalize, text="Normalize")
    notebook.add(tab_scaling, text="Scaling")
    notebook.add(tab_transform, text="Transformation")
    notebook.add(tab_encoding, text="Encoding")
    
    # Normalize tab
    ttk.Label(tab_normalize, text="Normalize numeric features:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', padx=10, pady=5)
    
    normalize_method = tk.StringVar()
    normalize_method.set("standard")
    
    ttk.Radiobutton(tab_normalize, text="Standard Scaling (Z-score)", variable=normalize_method, value="standard").pack(anchor='w', padx=20, pady=2)
    ttk.Radiobutton(tab_normalize, text="Min-Max Scaling", variable=normalize_method, value="minmax").pack(anchor='w', padx=20, pady=2)
    ttk.Radiobutton(tab_normalize, text="Robust Scaling (using quantiles)", variable=normalize_method, value="robust").pack(anchor='w', padx=20, pady=2)
    
    # Select columns frame
    normalize_cols_frame = ttk.Frame(tab_normalize)
    normalize_cols_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(normalize_cols_frame, text="Select columns to normalize:").pack(anchor='w')
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create a listbox for column selection
    normalize_cols_listbox = tk.Listbox(normalize_cols_frame, selectmode=tk.MULTIPLE, height=6)
    normalize_cols_listbox.pack(fill=tk.X, pady=5)
    
    # Add columns to listbox
    for col in numeric_cols:
        normalize_cols_listbox.insert(tk.END, col)
    
    # Select all columns by default
    for i in range(len(numeric_cols)):
        normalize_cols_listbox.selection_set(i)
    
    # Transformation tab
    ttk.Label(tab_transform, text="Transform features:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', padx=10, pady=5)
    
    transform_method = tk.StringVar()
    transform_method.set("none")
    
    ttk.Radiobutton(tab_transform, text="None", variable=transform_method, value="none").pack(anchor='w', padx=20, pady=2)
    ttk.Radiobutton(tab_transform, text="Log Transform", variable=transform_method, value="log").pack(anchor='w', padx=20, pady=2)
    ttk.Radiobutton(tab_transform, text="Square Root Transform", variable=transform_method, value="sqrt").pack(anchor='w', padx=20, pady=2)
    ttk.Radiobutton(tab_transform, text="Box-Cox Transform (for positive values)", variable=transform_method, value="boxcox").pack(anchor='w', padx=20, pady=2)
    
    # Select columns for transformation
    transform_cols_frame = ttk.Frame(tab_transform)
    transform_cols_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(transform_cols_frame, text="Select columns to transform:").pack(anchor='w')
    
    # Create a listbox for column selection
    transform_cols_listbox = tk.Listbox(transform_cols_frame, selectmode=tk.MULTIPLE, height=6)
    transform_cols_listbox.pack(fill=tk.X, pady=5)
    
    # Add columns to listbox
    for col in numeric_cols:
        transform_cols_listbox.insert(tk.END, col)
    
    # Encoding tab
    ttk.Label(tab_encoding, text="Encode categorical features:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', padx=10, pady=5)
    
    encoding_method = tk.StringVar()
    encoding_method.set("onehot")
    
    ttk.Radiobutton(tab_encoding, text="One-Hot Encoding", variable=encoding_method, value="onehot").pack(anchor='w', padx=20, pady=2)
    ttk.Radiobutton(tab_encoding, text="Label Encoding", variable=encoding_method, value="label").pack(anchor='w', padx=20, pady=2)
    
    # Get categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Select columns for encoding
    encoding_cols_frame = ttk.Frame(tab_encoding)
    encoding_cols_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(encoding_cols_frame, text="Select categorical columns to encode:").pack(anchor='w')
    
    # Create a listbox for column selection
    encoding_cols_listbox = tk.Listbox(encoding_cols_frame, selectmode=tk.MULTIPLE, height=6)
    encoding_cols_listbox.pack(fill=tk.X, pady=5)
    
    # Add columns to listbox
    for col in categorical_cols:
        encoding_cols_listbox.insert(tk.END, col)
    
    # Select all columns by default
    for i in range(len(categorical_cols)):
        encoding_cols_listbox.selection_set(i)
    
    # Scaling tab - Range scaling options
    ttk.Label(tab_scaling, text="Scale features to a specific range:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', padx=10, pady=5)
    
    # Min-max fields
    range_frame = ttk.Frame(tab_scaling)
    range_frame.pack(fill=tk.X, padx=20, pady=5)
    
    ttk.Label(range_frame, text="Min:").grid(row=0, column=0, padx=5, pady=5)
    min_val = tk.StringVar(value="0")
    min_entry = ttk.Entry(range_frame, textvariable=min_val, width=10)
    min_entry.grid(row=0, column=1, padx=5, pady=5)
    
    ttk.Label(range_frame, text="Max:").grid(row=0, column=2, padx=5, pady=5)
    max_val = tk.StringVar(value="1")
    max_entry = ttk.Entry(range_frame, textvariable=max_val, width=10)
    max_entry.grid(row=0, column=3, padx=5, pady=5)
    
    # Select columns for scaling
    scaling_cols_frame = ttk.Frame(tab_scaling)
    scaling_cols_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(scaling_cols_frame, text="Select columns to scale:").pack(anchor='w')
    
    # Create a listbox for column selection
    scaling_cols_listbox = tk.Listbox(scaling_cols_frame, selectmode=tk.MULTIPLE, height=6)
    scaling_cols_listbox.pack(fill=tk.X, pady=5)
    
    # Add columns to listbox
    for col in numeric_cols:
        scaling_cols_listbox.insert(tk.END, col)
    
    # Function to apply preprocessing
    def apply_preprocessing():
        try:
            df = data.copy()
            
            # Create progress window
            progress_window, progress_bar, progress_label = show_progress_bar(
                dialog, 
                "Applying Preprocessing",
                max_value=4  # 4 steps: normalize, transform, scale, encode
            )
            
            # Apply normalization
            if normalize_method.get() != "none":
                # Get selected columns
                selected_indices = normalize_cols_listbox.curselection()
                cols_to_normalize = [numeric_cols[i] for i in selected_indices]
                
                progress_label.config(text="Normalizing data...")
                progress_window.update()
                
                if cols_to_normalize:
                    if normalize_method.get() == "standard":
                        scaler = preprocessing.StandardScaler()
                        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
                    elif normalize_method.get() == "minmax":
                        scaler = preprocessing.MinMaxScaler()
                        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
                    elif normalize_method.get() == "robust":
                        scaler = preprocessing.RobustScaler()
                        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
                    
                    # Round float values after normalization
                    df = round_floats(df)
            
            progress_bar["value"] = 1
            progress_window.update()
            
            # Apply transformation
            if transform_method.get() != "none":
                # Get selected columns
                selected_indices = transform_cols_listbox.curselection()
                cols_to_transform = [numeric_cols[i] for i in selected_indices]
                
                progress_label.config(text="Transforming data...")
                progress_window.update()
                
                if cols_to_transform:
                    for col in cols_to_transform:
                        # Ensure we don't have negative values for certain transforms
                        data_col = df[col]
                        
                        if transform_method.get() == "log":
                            # Shift data to positive if needed for log transform
                            if data_col.min() <= 0:
                                shift = abs(data_col.min()) + 1
                                df[col] = np.log(data_col + shift)
                            else:
                                df[col] = np.log(data_col)
                        
                        elif transform_method.get() == "sqrt":
                            # Shift data to positive if needed for sqrt transform
                            if data_col.min() < 0:
                                shift = abs(data_col.min())
                                df[col] = np.sqrt(data_col + shift)
                            else:
                                df[col] = np.sqrt(data_col)
                        
                        elif transform_method.get() == "boxcox":
                            # Box-Cox requires positive data
                            if data_col.min() <= 0:
                                shift = abs(data_col.min()) + 1
                                from scipy import stats
                                df[col], _ = stats.boxcox(data_col + shift)
                            else:
                                from scipy import stats
                                df[col], _ = stats.boxcox(data_col)
                    
                    # Round float values after transformation
                    df = round_floats(df)
            
            progress_bar["value"] = 2
            progress_window.update()
            
            # Apply scaling to specific range
            if min_val.get() and max_val.get():
                try:
                    min_range = float(min_val.get())
                    max_range = float(max_val.get())
                    
                    # Get selected columns
                    selected_indices = scaling_cols_listbox.curselection()
                    cols_to_scale = [numeric_cols[i] for i in selected_indices]
                    
                    progress_label.config(text="Scaling data...")
                    progress_window.update()
                    
                    if cols_to_scale:
                        scaler = preprocessing.MinMaxScaler(feature_range=(min_range, max_range))
                        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                        
                        # Round float values after scaling
                        df = round_floats(df)
                except ValueError:
                    messagebox.showerror("Error", "Min and Max values must be numbers.")
            
            progress_bar["value"] = 3
            progress_window.update()
            
            # Apply encoding
            if categorical_cols:
                # Get selected columns
                selected_indices = encoding_cols_listbox.curselection()
                cols_to_encode = [categorical_cols[i] for i in selected_indices]
                
                progress_label.config(text="Encoding categorical data...")
                progress_window.update()
                
                if cols_to_encode:
                    if encoding_method.get() == "onehot":
                        # One-hot encoding
                        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
                    elif encoding_method.get() == "label":
                        # Label encoding
                        le = preprocessing.LabelEncoder()
                        for col in cols_to_encode:
                            df[col] = le.fit_transform(df[col].astype(str))
            
            progress_bar["value"] = 4
            progress_label.config(text="Preprocessing complete!")
            progress_window.update()
            
            # Short pause to show completion
            time.sleep(0.5)
            progress_window.destroy()
            
            # Final rounding before returning
            df = round_floats(df)
            return_data[0] = df
            dialog.destroy()
            messagebox.showinfo("Success", "Preprocessing applied successfully!")
            
        except Exception as e:
            if 'progress_window' in locals() and progress_window:
                progress_window.destroy()
            messagebox.showerror("Error", f"Error applying preprocessing: {str(e)}")
    
    # Function to cancel
    def cancel():
        dialog.destroy()
    
    # Buttons frame
    buttons_frame = ttk.Frame(dialog)
    buttons_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Button(buttons_frame, text="Apply", command=apply_preprocessing).pack(side=tk.RIGHT, padx=5)
    ttk.Button(buttons_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=5)
    
    # Wait for the dialog to close
    dialog.wait_window()
    
    # Return the processed data or None if canceled
    return return_data[0]

def show_progress_bar(parent, title="Processing", max_value=100):
    """
    Create and return a progress bar dialog
    
    Args:
        parent: Parent window
        title: Title of the progress window
        max_value: Maximum value for the progress bar
        
    Returns:
        tuple: (progress_window, progress_bar, progress_label)
    """
    progress_window = tk.Toplevel(parent)
    progress_window.title(title)
    progress_window.geometry("400x150")
    progress_window.transient(parent)
    progress_window.grab_set()
    
    # Center the window
    progress_window.update_idletasks()
    width = progress_window.winfo_width()
    height = progress_window.winfo_height()
    x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
    y = (progress_window.winfo_screenheight() // 2) - (height // 2)
    progress_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Create and pack a label
    progress_label = ttk.Label(progress_window, text="Initializing...", font=('TkDefaultFont', 10))
    progress_label.pack(pady=(20, 10))
    
    # Create and pack a progressbar
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate", maximum=max_value)
    progress_bar.pack(pady=10, padx=20)
    
    # Create a cancel button
    cancel_button = ttk.Button(progress_window, text="Cancel", command=progress_window.destroy)
    cancel_button.pack(pady=10)
    
    # Force update the window
    progress_window.update()
    
    return progress_window, progress_bar, progress_label

# Additional helper function to display data statistics
def show_data_statistics(data):
    """
    Show statistical information about the dataset
    """
    # Create a new window
    stats_window = tk.Toplevel()
    stats_window.title("Data Statistics")
    stats_window.geometry("800x600")
    
    # Create a notebook for tabs
    notebook = ttk.Notebook(stats_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create tabs
    tab_summary = ttk.Frame(notebook)
    tab_correlation = ttk.Frame(notebook)
    tab_distribution = ttk.Frame(notebook)
    
    notebook.add(tab_summary, text="Summary")
    notebook.add(tab_correlation, text="Correlation")
    notebook.add(tab_distribution, text="Distribution")
    
    # Summary tab
    summary_frame = ttk.Frame(tab_summary)
    summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Get summary statistics
    desc_stats = data.describe(include='all').transpose()
    desc_stats = desc_stats.round(6)  # Round to 6 decimal places instead of 3
    
    # Add column with data types
    desc_stats['dtype'] = data.dtypes
    
    # Add column with missing values count
    desc_stats['missing'] = data.isnull().sum()
    
    # Create a treeview to display statistics
    tree = ttk.Treeview(summary_frame)
    tree["columns"] = list(desc_stats.columns)
    tree["show"] = "headings"
    
    # Set column headings
    for col in desc_stats.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    # Insert data rows
    for i, row in desc_stats.iterrows():
        values = list(row)
        tree.insert("", tk.END, text=i, values=values, iid=i)
    
    # Add a scrollbar
    tree_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(fill=tk.BOTH, expand=True)
    
    # Correlation tab
    correlation_frame = ttk.Frame(tab_correlation)
    correlation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Only compute correlation for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if not numeric_data.empty:
        # Compute correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create a figure for correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        cax = ax.matshow(corr_matrix, cmap='coolwarm')
        fig.colorbar(cax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=90)
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add the correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax.text(i, j, round(corr_matrix.iloc[j, i], 2),
                        ha="center", va="center", color="black")
        
        # Add title
        plt.title('Correlation Matrix')
        
        # Embed the plot
        canvas = FigureCanvasTkAgg(fig, master=correlation_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        ttk.Label(correlation_frame, text="No numeric columns available for correlation analysis").pack(pady=20)
    
    # Distribution tab
    distribution_frame = ttk.Frame(tab_distribution)
    distribution_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Control frame for column selection
    control_frame = ttk.Frame(distribution_frame)
    control_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(control_frame, text="Select column:").pack(side=tk.LEFT, padx=5)
    
    column_var = tk.StringVar()
    if not data.columns.empty:
        column_var.set(data.columns[0])
    
    column_dropdown = ttk.Combobox(control_frame, textvariable=column_var, values=list(data.columns))
    column_dropdown.pack(side=tk.LEFT, padx=5)
    
    # Frame for the plot
    plot_frame = ttk.Frame(distribution_frame)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Function to update the distribution plot
    def update_distribution_plot():
        # Clear the plot frame
        for widget in plot_frame.winfo_children():
            widget.destroy()
        
        selected_column = column_var.get()
        
        if selected_column:
            column_data = data[selected_column]
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Check if numeric or categorical
            if column_data.dtype.kind in 'ifc':  # numeric
                # Histogram
                ax.hist(column_data.dropna(), bins=30, alpha=0.7)
                ax.set_title(f'Distribution of {selected_column}')
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Frequency')
                
                # Add a second axis for the density plot
                ax2 = ax.twinx()
                column_data.plot.kde(ax=ax2, color='red')
                ax2.set_ylabel('Density')
            else:  # categorical
                # Bar chart for value counts
                value_counts = column_data.value_counts()
                value_counts.plot.bar(ax=ax)
                ax.set_title(f'Value Counts for {selected_column}')
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
            
            # Embed the plot
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Update plot button
    update_button = ttk.Button(control_frame, text="Update Plot", command=update_distribution_plot)
    update_button.pack(side=tk.LEFT, padx=10)
    
    # Initial plot
    if not data.columns.empty:
        update_distribution_plot()
    
    # Close button
    close_button = ttk.Button(stats_window, text="Close", command=stats_window.destroy)
    close_button.pack(pady=10)
