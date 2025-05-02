import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

def create_cluster_plot(data, cluster_column='cluster', method='kmeans', figsize=(10, 6), show_plot=True, return_figure=True):
    """
    Create a visualization of clustered data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data with a cluster column indicating the cluster assignments
    cluster_column : str, default='cluster'
        Name of the column that contains cluster assignments
    method : str, default='kmeans'
        The clustering method used, for plot title
    figsize : tuple, default=(10, 6)
        Figure size
    show_plot : bool, default=True
        Whether to display the plot immediately
    return_figure : bool, default=True
        Whether to return the figure object
    """
    if cluster_column not in data.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in the data")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Get numeric columns for dimensionality reduction
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the cluster column from numeric columns if it's there
    if cluster_column in numeric_cols:
        numeric_cols.remove(cluster_column)
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for plotting")
    
    # Get unique clusters
    clusters = df[cluster_column].unique()
    n_clusters = len(clusters)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Choose dimensionality reduction based on data size
    if len(numeric_cols) > 2:
        # If more than 50 samples, use PCA which is faster
        if len(df) > 50:
            reducer = PCA(n_components=2)
            reducer_name = "PCA"
        else:
            # t-SNE works better for smaller datasets
            reducer = TSNE(n_components=2, random_state=42)
            reducer_name = "t-SNE"
        
        # Apply dimensionality reduction
        reduced_data = reducer.fit_transform(df[numeric_cols])
        
        # Add reduced dimensions to dataframe
        df['x'] = reduced_data[:, 0]
        df['y'] = reduced_data[:, 1]
        
        plot_title = f"{method.upper()} Clustering Results ({reducer_name} projection)"
    else:
        # Use the first two numeric columns directly
        df['x'] = df[numeric_cols[0]]
        df['y'] = df[numeric_cols[1]] if len(numeric_cols) > 1 else df[numeric_cols[0]]
        plot_title = f"{method.upper()} Clustering Results"
    
    # Create a scatter plot for each cluster
    ax = fig.add_subplot(111)
    
    # Use categorical colormap for clusters
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    for i, cluster in enumerate(clusters):
        cluster_data = df[df[cluster_column] == cluster]
        ax.scatter(
            cluster_data['x'], 
            cluster_data['y'], 
            s=50, 
            c=[cmap(i)], 
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    
    # Add labels and legend
    ax.set_title(plot_title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    if return_figure:
        return fig
    return None

def create_kmeans_plot(data, cluster_column='cluster', show_plot=True):
    """Create a visualization of K-means clustering results."""
    return create_cluster_plot(data, cluster_column, method='K-means', show_plot=show_plot)

def create_pam_plot(data, cluster_column='cluster', show_plot=True):
    """Create a visualization of PAM clustering results."""
    return create_cluster_plot(data, cluster_column, method='PAM', show_plot=show_plot)

def create_dbscan_plot(data, cluster_column='cluster', show_plot=True):
    """Create a visualization of DBSCAN clustering results."""
    return create_cluster_plot(data, cluster_column, method='DBSCAN', show_plot=show_plot)

def display_plot_in_window(fig, title="Clustering Visualization", parent=None):
    """
    Display a matplotlib figure in a maximizable Tkinter window.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to display
    title : str, default="Clustering Visualization"
        The window title
    parent : tkinter.Tk or tkinter.Toplevel, optional
        The parent window
    """
    plot_window = tk.Toplevel(parent) if parent else tk.Tk()
    plot_window.title(title)
    plot_window.geometry("800x600")
    
    # Allow the window to be resized
    plot_window.resizable(True, True)
    
    # Create main frame
    main_frame = ttk.Frame(plot_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create canvas for the plot
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add toolbar for zooming, panning, etc.
    toolbar_frame = ttk.Frame(plot_window)
    toolbar_frame.pack(fill=tk.X, padx=10)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    # Add button frame at the bottom
    button_frame = ttk.Frame(plot_window)
    button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    # Add maximize button and other controls
    def maximize_window():
        plot_window.attributes('-zoomed', True)
    
    def restore_window():
        plot_window.attributes('-zoomed', False)
    
    ttk.Button(button_frame, text="Maximize", command=maximize_window).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_frame, text="Restore", command=restore_window).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_frame, text="Close", command=plot_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    # Run the window if it's not a child window
    if not parent:
        plot_window.mainloop()
