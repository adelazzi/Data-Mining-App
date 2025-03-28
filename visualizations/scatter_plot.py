import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_scatter_plot(data):
    """
    Create a scatter plot visualization for the given dataset
    """
    # Create a new window
    plot_window = tk.Toplevel()
    plot_window.title("Scatter Plot")
    plot_window.geometry("800x600")
    
    # Create frame for column selection
    control_frame = ttk.Frame(plot_window)
    control_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Only show numeric columns for plotting
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        ttk.Label(control_frame, text="Need at least 2 numeric columns for scatter plot").pack()
        return
    
    # X-axis selection
    ttk.Label(control_frame, text="X-axis:").pack(side=tk.LEFT, padx=(0, 5))
    x_var = tk.StringVar()
    x_dropdown = ttk.Combobox(control_frame, textvariable=x_var, values=numeric_columns, width=15)
    x_dropdown.pack(side=tk.LEFT, padx=(0, 10))
    x_dropdown.current(0)
    
    # Y-axis selection
    ttk.Label(control_frame, text="Y-axis:").pack(side=tk.LEFT, padx=(0, 5))
    y_var = tk.StringVar()
    y_dropdown = ttk.Combobox(control_frame, textvariable=y_var, values=numeric_columns, width=15)
    y_dropdown.pack(side=tk.LEFT, padx=(0, 10))
    if len(numeric_columns) > 1:
        y_dropdown.current(1)
    else:
        y_dropdown.current(0)
    
    # Color by selection (optional)
    ttk.Label(control_frame, text="Color by:").pack(side=tk.LEFT, padx=(0, 5))
    color_var = tk.StringVar()
    color_values = ["None"] + data.columns.tolist()
    color_dropdown = ttk.Combobox(control_frame, textvariable=color_var, values=color_values, width=15)
    color_dropdown.pack(side=tk.LEFT)
    color_dropdown.current(0)
    
    # Create frame for the plot
    plot_frame = ttk.Frame(plot_window)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Function to update the plot
    def update_plot():
        # Clear the plot frame
        for widget in plot_frame.winfo_children():
            widget.destroy()
        
        # Get selected columns
        x_col = x_var.get()
        y_col = y_var.get()
        color_col = color_var.get()
        
        if not x_col or not y_col:
            return
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        if color_col and color_col != "None":
            scatter = ax.scatter(data[x_col], data[y_col], c=data[color_col].astype('category').cat.codes, 
                       alpha=0.7, cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title=color_col)
            ax.add_artist(legend1)
        else:
            ax.scatter(data[x_col], data[y_col], alpha=0.7)
        
        # Set title and labels
        ax.set_title(f'Scatter Plot: {y_col} vs {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Button to update the plot
    update_button = ttk.Button(control_frame, text="Update Plot", command=update_plot)
    update_button.pack(side=tk.LEFT, padx=10)
    
    # Initial plot
    update_plot()
