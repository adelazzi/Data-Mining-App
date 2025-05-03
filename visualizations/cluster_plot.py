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

def create_agnes_plot(data, cluster_column='cluster', show_plot=True):
    """Create a visualization of AGNES (Agglomerative Hierarchical Clustering) results."""
    return create_cluster_plot(data, cluster_column, method='AGNES', show_plot=show_plot)

def create_diana_plot(data, cluster_column='cluster', show_plot=True):
    """Create a visualization of DIANA (Divisive Analysis) clustering results."""
    return create_cluster_plot(data, cluster_column, method='DIANA', show_plot=show_plot)

def create_agnes_dendrogram(data, method='ward', affinity='euclidean', show_plot=True):
    """
    Create a modern, professional dendrogram visualization for hierarchical clustering.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to create dendrogram for
    method : str, default='ward'
        Linkage method to use: 'ward', 'complete', 'average', 'single'
    affinity : str, default='euclidean'
        Distance metric to use
    show_plot : bool, default=True
        Whether to display the plot immediately
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler
    
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Modern color palette
    colors = {
        'background': '#FFFFFF',
        'text': '#333333',
        'grid': '#EEEEEE',
        'line': '#505050',
        'highlight': '#2980b9',
        'accent': '#3498db',
        'dendrogram': '#2c3e50'
    }
    
    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Create linkage matrix using scipy's hierarchical clustering
    if method == 'ward' and affinity != 'euclidean':
        print("Warning: Ward linkage only works with Euclidean distance. Using Euclidean.")
        affinity = 'euclidean'
    
    # Compute the linkage matrix
    Z = linkage(scaled_data, method=method, metric=affinity if method != 'ward' else 'euclidean')
    
    # Create figure for the dendrogram with high-resolution and modern aspect
    fig = plt.figure(figsize=(14, 8), dpi=100, facecolor=colors['background'])
    
    # Create a single subplot with specific style
    ax = fig.add_subplot(111, facecolor=colors['background'])
    
    # Set background color
    fig.patch.set_facecolor(colors['background'])
    
    # Create the dendrogram with modern colors
    dendrogram(
        Z,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0.7*max(Z[:,2]),  # Color threshold for better visualization
        above_threshold_color=colors['line'],
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        no_labels=False if len(data) < 50 else True,  # Hide labels for large datasets
        labels=None if len(data) >= 50 else [f"Sample {i}" for i in range(len(data))],
    )
    
    # Add title and labels with modern typography
    ax.set_title(f'Hierarchical Clustering Dendrogram\n{method.capitalize()} Linkage / {affinity.capitalize()} Distance', 
                fontsize=16, 
                color=colors['text'],
                fontweight='bold',
                pad=20)
    
    ax.set_xlabel('Samples', fontsize=14, color=colors['text'], labelpad=15)
    ax.set_ylabel('Distance', fontsize=14, color=colors['text'], labelpad=15)
    
    # Add annotations for better interpretation
    if len(data) < 100:  # Only for reasonably sized datasets
        largest_dist = max(Z[:,2])
        ax.axhline(y=0.7*largest_dist, c=colors['highlight'], linestyle='--', alpha=0.7)
        ax.text(len(data)/2, 0.71*largest_dist, 
                'Recommended\nClustering Level', 
                ha='center', va='bottom', 
                color=colors['highlight'], 
                fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add legend in modern style
    legend_elements = [
        plt.Line2D([0], [0], color=colors['dendrogram'], lw=4, label='Cluster Branch'),
        plt.Line2D([0], [0], color=colors['highlight'], lw=2, linestyle='--', label='Recommended Cut')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    # Style the spines (edges)
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
        spine.set_linewidth(0.5)
    
    # Style the grid
    ax.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Style the ticks
    ax.tick_params(colors=colors['text'], labelsize=10)
    
    # Optimize layout
    plt.tight_layout()
    
    # Add watermark or label in bottom corner
    fig.text(0.98, 0.02, 'Hierarchical Clustering Analysis', 
            fontsize=8, color=colors['text'], alpha=0.5, ha='right')
    
    # Add interpretation note
    interpretation_text = (
        "Interpretation: The dendrogram shows hierarchical relationships between clusters.\n"
        f"Clusters are formed using {method} linkage criterion with {affinity} distance metric.\n"
        "The height (y-axis) represents the distance at which clusters are merged."
    )
    
    fig.text(0.02, 0.02, interpretation_text, 
             fontsize=9, color=colors['text'], alpha=0.7, ha='left',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    if show_plot:
        plt.show()
    
    return fig

def create_diana_dendrogram(data, show_plot=True):
    """
    Create a modern, professional dendrogram visualization for hierarchical clustering using DIANA.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to create dendrogram for
    show_plot : bool, default=True
        Whether to display the plot immediately
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.preprocessing import StandardScaler
    
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Modern color palette
    colors = {
        'background': '#FFFFFF',
        'text': '#333333',
        'grid': '#EEEEEE',
        'line': '#505050',
        'highlight': '#2980b9',
        'accent': '#3498db',
        'dendrogram': '#2c3e50'
    }
    
    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Create DIANA method for hierarchical clustering (or use divisive if available)
    def divisive(data):
        '''
        Perform the DIANA (Divisive Analysis) algorithm for hierarchical clustering.

        Arguments
        ----------
        data : pandas.DataFrame
            The dataset for clustering

        Returns
        -------
        Z : numpy.ndarray
            The linkage matrix for hierarchical clustering
        '''
        from diana_clustering.distance_matrix import DistanceMatrix

        # Create the similarity matrix using the DistanceMatrix class
        similarity_matrix = DistanceMatrix(data)
        n_samples = data.shape[0]

        # Initial cluster is the entire dataset
        clusters = [list(range(n_samples))]
        
        # Create an empty list to store the linkage matrix
        linkage_matrix = []
        
        while len(clusters) < n_samples:
            # Calculate the diameters (max distance) for each cluster
            cluster_diameters = [np.max(similarity_matrix[cluster][:, cluster]) for cluster in clusters]
            
            # Find the cluster with the maximum diameter
            max_cluster_idx = np.argmax(cluster_diameters)
            
            # Find the element with the maximum distance to its other members (most different)
            max_difference_idx = np.argmax(np.mean(similarity_matrix[clusters[max_cluster_idx]][:, clusters[max_cluster_idx]], axis=1))
            
            # Create the "splinter" group, starting with the most different element
            splinter_group = [clusters[max_cluster_idx][max_difference_idx]]
            
            # Remove the chosen element from the current cluster
            last_cluster = clusters[max_cluster_idx]
            del last_cluster[max_difference_idx]
            
            # Now, attempt to further split the cluster by iterating over the remaining elements
            while True:
                split = False
                for j in range(len(last_cluster) - 1, -1, -1):
                    # Calculate distances from the current element to the splinter group and the remaining cluster
                    splinter_distances = similarity_matrix[last_cluster[j], splinter_group]
                    remaining_distances = similarity_matrix[last_cluster[j], np.delete(last_cluster, j, axis=0)]
                    
                    # If the average distance to the splinter group is smaller than to the rest, move it to the splinter group
                    if np.mean(splinter_distances) <= np.mean(remaining_distances):
                        splinter_group.append(last_cluster[j])
                        del last_cluster[j]
                        split = True
                        break
                
                if not split:
                    break
            
            # Update the list of clusters: replace the largest cluster with the new splinter group and the rest of the elements
            del clusters[max_cluster_idx]
            clusters.append(splinter_group)
            clusters.append(last_cluster)
            
            # Record the split in the linkage matrix (format: [cluster1, cluster2, distance, num_elements])
            dist = np.max(similarity_matrix[splinter_group][:, splinter_group])
            linkage_matrix.append([len(clusters) - 2, len(clusters) - 1, dist, len(splinter_group) + len(last_cluster)])
        
        # Convert the linkage matrix to a numpy array for compatibility with scipy dendrogram function
        Z = np.array(linkage_matrix)
        
        return Z

    Z = divisive(scaled_data)

    # Create figure for the dendrogram with high-resolution and modern aspect
    fig = plt.figure(figsize=(14, 8), dpi=100, facecolor=colors['background'])
    
    # Create a single subplot with specific style
    ax = fig.add_subplot(111, facecolor=colors['background'])
    
    # Set background color
    fig.patch.set_facecolor(colors['background'])
    
    # Create the dendrogram with modern colors
    dendrogram(
        Z,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0.7*max(Z[:,2]),  # Color threshold for better visualization
        above_threshold_color=colors['line'],
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        no_labels=False if len(data) < 50 else True,  # Hide labels for large datasets
        labels=None if len(data) >= 50 else [f"Sample {i}" for i in range(len(data))],
    )
    
    # Add title and labels with modern typography
    ax.set_title(f'Divisive Hierarchical Clustering Dendrogram',
                fontsize=16, 
                color=colors['text'],
                fontweight='bold',
                pad=20)
    
    ax.set_xlabel('Samples', fontsize=14, color=colors['text'], labelpad=15)
    ax.set_ylabel('Distance', fontsize=14, color=colors['text'], labelpad=15)
    
    # Add annotations for better interpretation
    if len(data) < 100:  # Only for reasonably sized datasets
        largest_dist = max(Z[:,2])
        ax.axhline(y=0.7*largest_dist, c=colors['highlight'], linestyle='--', alpha=0.7)
        ax.text(len(data)/2, 0.71*largest_dist, 
                'Recommended\nClustering Level', 
                ha='center', va='bottom', 
                color=colors['highlight'], 
                fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add legend in modern style
    legend_elements = [
        plt.Line2D([0], [0], color=colors['dendrogram'], lw=4, label='Cluster Branch'),
        plt.Line2D([0], [0], color=colors['highlight'], lw=2, linestyle='--', label='Recommended Cut')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    # Style the spines (edges)
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
        spine.set_linewidth(0.5)
    
    # Style the grid
    ax.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Style the ticks
    ax.tick_params(colors=colors['text'], labelsize=10)
    
    # Optimize layout
    plt.tight_layout()
    
    # Add watermark or label in bottom corner
    fig.text(0.98, 0.02, 'Hierarchical Clustering Analysis', 
            fontsize=8, color=colors['text'], alpha=0.5, ha='right')
    
    # Add interpretation note
    interpretation_text = (
        "Interpretation: The dendrogram shows hierarchical relationships between clusters.\n"
        "The height (y-axis) represents the distance at which clusters are merged."
    )
    
    fig.text(0.02, 0.02, interpretation_text, 
             fontsize=9, color=colors['text'], alpha=0.7, ha='left',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    if show_plot:
        plt.show()
    
    return fig

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
