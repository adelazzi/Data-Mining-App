import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

def create_box_plot(data):
    """
    Create a box plot visualization for the given dataset
    """
    # Create a new window
    plot_window = tk.Toplevel()
    plot_window.title("Box Plot")
    plot_window.geometry("800x600")
    
    # Create frame for column selection
    control_frame = ttk.Frame(plot_window)
    control_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Get all columns for plotting
    all_columns = data.columns.tolist()
    
    if not all_columns:
        ttk.Label(control_frame, text="No columns available for plotting").pack()
        return
    
    ttk.Label(control_frame, text="Select columns:").pack(side=tk.LEFT, padx=(0, 10))
    
    # Create listbox for column selection with multiple selection enabled
    column_listbox = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=4)
    column_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Populate listbox with all column names
    for col in all_columns:
        column_listbox.insert(tk.END, col)
    
    # Select first column by default
    column_listbox.selection_set(0)
    
    # Create frame for the plot
    plot_frame = ttk.Frame(plot_window)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Function to update the plot
    def update_plot():
        # Clear the plot frame
        for widget in plot_frame.winfo_children():
            widget.destroy()
        
        # Get selected columns
        selected_indices = column_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Info", "Please select at least one column")
            return
        
        selected_columns = [all_columns[i] for i in selected_indices]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate numeric and categorical columns
        selected_data = data[selected_columns].copy()
        numeric_cols = selected_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in selected_columns if col not in numeric_cols]
        
        # Create box plot for numeric data
        if numeric_cols:
            selected_data[numeric_cols].boxplot(ax=ax)
        
        # Handle categorical data if present
        if categorical_cols:
            # Convert categorical columns to category codes for plotting
            for col in categorical_cols:
                # Get value counts for the categorical column
                value_counts = selected_data[col].value_counts().sort_index()
                
                # Display separate bar chart for categorical data
                if len(categorical_cols) == 1 and not numeric_cols:
                    # If only categorical data is selected, show a bar chart
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'Value Distribution for {col}')
                    ax.set_ylabel('Count')
                else:
                    # If mixed with numeric data, create a separate plot
                    value_counts_ax = ax.twinx()
                    value_counts.plot(kind='bar', ax=value_counts_ax, alpha=0.5, color='green')
                    value_counts_ax.set_ylabel('Categorical Count', color='green')
                    ax.set_title('Box Plot (Numeric) and Value Distribution (Categorical)')
        
        # Set title and labels
        if numeric_cols and not categorical_cols:
            ax.set_title('Box Plot')
            ax.set_xlabel('Features')
            ax.set_ylabel('Values')
            
        # Adjust layout
        plt.tight_layout()
        
        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Button to update the plot
    update_button = ttk.Button(control_frame, text="Update Plot", command=update_plot)
    update_button.pack(side=tk.LEFT, padx=10)
    
    # Initial plot
    update_plot()
