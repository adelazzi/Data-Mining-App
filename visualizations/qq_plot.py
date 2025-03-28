import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.stats as stats
import pandas as pd

def create_qq_plot(data):
    """
    Create a QQ plot visualization for the given dataset
    """
    # Create a new window
    plot_window = tk.Toplevel()
    plot_window.title("QQ Plot")
    plot_window.geometry("800x600")
    
    # Create frame for column selection
    control_frame = ttk.Frame(plot_window)
    control_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Get all columns
    all_columns = data.columns.tolist()
    
    if not all_columns:
        ttk.Label(control_frame, text="No columns available for plotting").pack()
        return
    
    ttk.Label(control_frame, text="Select column:").pack(side=tk.LEFT, padx=(0, 10))
    
    # Create dropdown for column selection
    selected_column = tk.StringVar()
    column_dropdown = ttk.Combobox(control_frame, textvariable=selected_column, values=all_columns)
    column_dropdown.pack(side=tk.LEFT)
    
    # Select first column by default
    if all_columns:
        column_dropdown.current(0)
    
    # Create frame for the plot
    plot_frame = ttk.Frame(plot_window)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Function to update the plot
    def update_plot():
        # Clear the plot frame
        for widget in plot_frame.winfo_children():
            widget.destroy()
        
        # Get selected column
        col = selected_column.get()
        if not col:
            messagebox.showinfo("Info", "Please select a column")
            return
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data[col]):
            # Create QQ plot for numeric data
            data_values = data[col].dropna()
            stats.probplot(data_values, dist="norm", plot=ax)
            ax.set_title(f'QQ Plot - {col}')
        else:
            # For categorical data, show value distribution
            value_counts = data[col].value_counts().sort_index()
            
            # Create bar chart for categorical data
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Value Distribution for {col} (Categorical)')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            
            # Add informational text
            ax.annotate('Note: QQ plots are only for numeric data.\nShowing value distribution instead.',
                        xy=(0.5, 0.9), xycoords='axes fraction',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
            
            plt.xticks(rotation=45)
        
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
