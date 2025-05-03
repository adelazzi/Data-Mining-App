import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform, cdist
from visualizations.cluster_plot import create_kmeans_plot, create_pam_plot, create_dbscan_plot, create_agnes_plot, create_diana_plot

def calculate_intra_cluster_distance(data, labels, centroids=None):
    """
    Calculate the average intra-cluster distance (cohesion).
    Lower values indicate more compact clusters.
    
    Parameters:
    -----------
    data : array-like
        The input data
    labels : array-like
        Cluster labels for each point
    centroids : array-like, optional
        Cluster centroids. If None, centroids will be calculated
        
    Returns:
    --------
    float
        Average intra-cluster distance
    """
    unique_labels = np.unique(labels)
    intra_distances = []
    
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) <= 1:
            # Skip clusters with only one point (intra-distance is 0)
            continue
            
        # Calculate pairwise distances within the cluster
        cluster_distances = pdist(cluster_points)
        intra_distances.append(np.mean(cluster_distances))
    
    # Return average of all intra-cluster distances
    if intra_distances:
        return np.mean(intra_distances)
    else:
        return 0.0

def calculate_inter_cluster_distance(data, labels, centroids=None):
    """
    Calculate the average inter-cluster distance (separation).
    Higher values indicate better separated clusters.
    
    Parameters:
    -----------
    data : array-like
        The input data
    labels : array-like
        Cluster labels for each point
    centroids : array-like, optional
        Cluster centroids. If None, centroids will be calculated
        
    Returns:
    --------
    float
        Average inter-cluster distance
    """
    unique_labels = np.unique(labels)
    
    # Calculate centroids if not provided
    if centroids is None:
        centroids = []
        for label in unique_labels:
            cluster_points = data[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        centroids = np.array(centroids)
    
    # Calculate pairwise distances between centroids
    if len(centroids) <= 1:
        return 0.0
    
    centroid_distances = pdist(centroids)
    return np.mean(centroid_distances)

def perform_kmeans_clustering(data, n_clusters=3, random_state=42, max_iter=300, n_init=10, visualize=True):
    """
    Perform K-means clustering on the given data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to cluster
    n_clusters : int, default=3
        Number of clusters
    random_state : int, default=42
        Random state for reproducibility
    max_iter : int, default=300
        Maximum number of iterations for the K-means algorithm
    n_init : int, default=10
        Number of times the K-means algorithm is run with different centroid seeds
    visualize : bool, default=True
        Whether to visualize the clustering results
        
    Returns:
    --------
    pandas.DataFrame
        The original data with an additional 'cluster' column
    dict
        Evaluation metrics
    """
    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=n_init)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the original data
    result_data = data.copy()
    result_data['cluster'] = clusters
    
    # Calculate evaluation metrics
    metrics = {}
    if len(data) > n_clusters:  # Metrics require more samples than clusters
        try:
            metrics['silhouette'] = round(silhouette_score(scaled_data, clusters), 3)
            metrics['intra_cluster'] = round(calculate_intra_cluster_distance(scaled_data, clusters), 3)
            metrics['inter_cluster'] = round(calculate_inter_cluster_distance(scaled_data, clusters, kmeans.cluster_centers_), 3)
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
    
    # Visualize if requested
    if visualize:
        create_kmeans_plot(result_data)
    
    return result_data, metrics

def perform_pam_clustering(data, n_clusters=3, random_state=42, max_iter=100, visualize=True):
    """
    Perform PAM (Partitioning Around Medoids) clustering on the given data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to cluster
    n_clusters : int, default=3
        Number of clusters
    random_state : int, default=42
        Random state for reproducibility
    max_iter : int, default=100
        Maximum number of iterations for the PAM algorithm
    visualize : bool, default=True
        Whether to visualize the clustering results
        
    Returns:
    --------
    pandas.DataFrame
        The original data with an additional 'cluster' column
    dict
        Evaluation metrics
    """
    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Custom PAM implementation
    n_samples = scaled_data.shape[0]
    
    # 1. Initialize medoids by selecting random points
    medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    medoids = scaled_data[medoid_indices]
    
    # 2. Assign each point to the nearest medoid
    distances = cdist(scaled_data, medoids)
    cluster_labels = np.argmin(distances, axis=1)
    
    # 3. For each cluster, find the point that minimizes the sum of distances to other points
    for _ in range(max_iter):
        old_medoid_indices = medoid_indices.copy()
        
        # For each cluster, update the medoid
        for cluster_idx in range(n_clusters):
            # Get all points in the current cluster
            cluster_points_indices = np.where(cluster_labels == cluster_idx)[0]
            
            if len(cluster_points_indices) == 0:
                continue  # Empty cluster, skip
            
            # Calculate pairwise distances for points in this cluster
            cluster_points = scaled_data[cluster_points_indices]
            
            # Find point with minimal sum of distances to other points in cluster
            if len(cluster_points) > 1:
                cluster_distances = cdist(cluster_points, cluster_points)
                min_distance_idx = np.argmin(np.sum(cluster_distances, axis=1))
                medoid_indices[cluster_idx] = cluster_points_indices[min_distance_idx]
        
        # If medoids didn't change, stop iterating
        if np.array_equal(old_medoid_indices, medoid_indices):
            break
        
        # Update medoids
        medoids = scaled_data[medoid_indices]
        
        # Reassign each point to the nearest medoid
        distances = cdist(scaled_data, medoids)
        cluster_labels = np.argmin(distances, axis=1)
    
    # Add cluster labels to the original data
    result_data = data.copy()
    result_data['cluster'] = cluster_labels
    
    # Calculate evaluation metrics
    metrics = {}
    if len(data) > n_clusters:  # Metrics require more samples than clusters
        try:
            metrics['silhouette'] = round(silhouette_score(scaled_data, cluster_labels), 3)
            metrics['intra_cluster'] = round(calculate_intra_cluster_distance(scaled_data, cluster_labels), 3)
            metrics['inter_cluster'] = round(calculate_inter_cluster_distance(scaled_data, cluster_labels, scaled_data[medoid_indices]), 3)
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
    
    # Visualize if requested
    if visualize:
        create_pam_plot(result_data)
    
    return result_data, metrics

def perform_dbscan_clustering(data, eps=0.5, min_samples=5, visualize=True):
    """
    Perform DBSCAN clustering on the given data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The data to cluster
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered a core point.
    visualize : bool, default=True
        Whether to visualize the clustering results

    Returns:
    --------
    pandas.DataFrame
        The original data with an additional 'cluster' column
    dict
        Evaluation metrics
    """
    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(scaled_data)

    # Add cluster labels to the original data
    result_data = data.copy()
    result_data['cluster'] = cluster_labels

    # Calculate evaluation metrics
    metrics = {}
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if n_clusters > 1:
        try:
            metrics['silhouette'] = round(silhouette_score(scaled_data, cluster_labels), 3)
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
    metrics['n_clusters'] = n_clusters
    metrics['n_noise_points'] = list(cluster_labels).count(-1)

    # Visualize if requested
    if visualize:
        create_dbscan_plot(result_data)

    return result_data, metrics

def perform_agnes_clustering(data, n_clusters=3, linkage='ward', affinity='euclidean', visualize=True):
    """
    Perform AGNES (Agglomerative Hierarchical Clustering) on the given data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to cluster
    n_clusters : int, default=3
        Number of clusters to form
    linkage : str, default='ward'
        Linkage criterion to use: 'ward', 'complete', 'average', 'single'
    affinity : str, default='euclidean'
        Metric used to compute the linkage: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'
    visualize : bool, default=True
        Whether to visualize the clustering results
        
    Returns:
    --------
    pandas.DataFrame
        The original data with an additional 'cluster' column
    dict
        Evaluation metrics
    """
    from sklearn.cluster import AgglomerativeClustering
    
    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Perform AGNES clustering
    # Note: 'ward' linkage only works with euclidean affinity
    if linkage == 'ward':
        agnes = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
    else:
        agnes = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity
        )
        
    cluster_labels = agnes.fit_predict(scaled_data)
    
    # Add cluster labels to the original data
    result_data = data.copy()
    result_data['cluster'] = cluster_labels
    
    # Calculate evaluation metrics
    metrics = {}
    if len(data) > n_clusters:  # Metrics require more samples than clusters
        try:
            metrics['silhouette'] = round(silhouette_score(scaled_data, cluster_labels), 3)
            metrics['intra_cluster'] = round(calculate_intra_cluster_distance(scaled_data, cluster_labels), 3)
            metrics['inter_cluster'] = round(calculate_inter_cluster_distance(scaled_data, cluster_labels), 3)
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
    
    # Store linkage and affinity in metrics for dendrogram visualization
    metrics['linkage'] = linkage
    metrics['affinity'] = affinity
    
    # Visualize if requested
    if visualize:
        create_agnes_plot(result_data)
    
    return result_data, metrics

def perform_diana_clustering(data, n_clusters=3, visualize=True):
    """
    Perform DIANA (Divisive Hierarchical Clustering) on the given data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to cluster
    n_clusters : int, default=3
        Number of clusters to form
    visualize : bool, default=True
        Whether to visualize the clustering results
        
    Returns:
    --------
    pandas.DataFrame
        The original data with an additional 'cluster' column
    dict
        Evaluation metrics
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from diana_clustering.algorithm import Diana

    # Get numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Perform DIANA clustering (using a third-party library or custom implementation)
    diana = Diana(n_clusters=n_clusters)
    cluster_labels = diana.fit(scaled_data)
    
    # Add cluster labels to the original data
    result_data = data.copy()
    result_data['cluster'] = cluster_labels
    
    # Calculate evaluation metrics
    metrics = {}
    if len(data) > n_clusters:  # Metrics require more samples than clusters
        try:
            metrics['silhouette'] = round(silhouette_score(scaled_data, cluster_labels), 3)
            metrics['intra_cluster'] = round(calculate_intra_cluster_distance(scaled_data, cluster_labels), 3)
            metrics['inter_cluster'] = round(calculate_inter_cluster_distance(scaled_data, cluster_labels), 3)
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
    
    # Visualize if requested (using custom visualization or dendrogram)
    if visualize:
        create_diana_plot(result_data)  # Create a plot for DIANA
    
    return result_data, metrics

def show_dendrogram(data, method='agnes', linkage='ward', affinity='euclidean'):
    """
    Show a dendrogram visualization for hierarchical clustering.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to visualize.
    method : str, default='agnes'
        The hierarchical clustering method: 'agnes' for agglomerative, 'diana' for divisive.
    linkage : str, default='ward'
        Linkage criterion to use for AGNES: 'ward', 'complete', 'average', 'single'.
        Ignored if method is 'diana'.
    affinity : str, default='euclidean'
        Metric used to compute the linkage. Ignored if method is 'diana'.
    """
    from visualizations.cluster_plot import create_agnes_dendrogram, display_plot_in_window
    
    try:
        if method == 'agnes':
            fig = create_agnes_dendrogram(data, method=linkage, affinity=affinity, show_plot=False)
            title = f'AGNES Dendrogram ({linkage} linkage)'
        #elif method == 'diana':
        #    fig = create_dendrogram(data, method='diana', show_plot=False)
        #    title = 'DIANA Dendrogram'
        else:
            raise ValueError(f"Unsupported hierarchical method: {method}")

        display_plot_in_window(fig, title=title)

    except Exception as e:
        import tkinter.messagebox as messagebox
        messagebox.showerror("Error", f"Failed to create dendrogram: {str(e)}")

def show_clustering_dialog(parent, data, method="kmeans"):
    """
    Show a dialog for clustering parameters and perform clustering.
    
    Parameters:
    -----------
    parent : tkinter.Tk
        Parent window
    data : pandas.DataFrame
        The data to cluster
    method : str, default="kmeans"
        Clustering method ("kmeans" or "pam")
        
    Returns:
    --------
    tuple
        (clustered_data, n_clusters, metrics) or None if canceled
    """
    # Create a new dialog window with a modern look
    dialog = tk.Toplevel(parent)
    method_name = 'K-means' if method == 'kmeans' else 'PAM' if method == 'pam' else 'DBSCAN' if method == 'dbscan' else 'DIANA' if method == 'diana' else 'AGNES'
    dialog.title(f"{method_name} Clustering")
    dialog.geometry("900x700")
    dialog.resizable(True, True)
    dialog.minsize(650, 500)
    
    # Define a more neutral color palette to match main application
    colors = {
        'background': '#f8f8f8',     # Light gray background
        'card': '#ffffff',           # White for cards/panels
        'primary': '#505050',        # Dark gray for primary elements
        'secondary': '#4a4a4a',      # Slightly darker for highlights
        'accent': '#333333',         # Dark gray for accents
        'text': '#333333',           # Dark gray for text
        'text_light': '#555555'      # Medium gray for secondary text
    }
    
    # Configure the dialog background
    dialog.configure(bg=colors['background'])
    
    # Track fullscreen state
    is_fullscreen = False
    
    # Add maximize/restore function
    def toggle_maximize():
        nonlocal is_fullscreen
        is_fullscreen = not is_fullscreen
        
        if is_fullscreen:
            # Save current position and size before maximizing
            dialog._geom_before_fullscreen = dialog.geometry()
            # Set to fullscreen or maximize based on platform
            w, h = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
            dialog.geometry(f"{w}x{h}+0+0")
            max_button.config(text="🗗")  # Change icon to restore
        else:
            # Restore to previous size
            if hasattr(dialog, '_geom_before_fullscreen'):
                dialog.geometry(dialog._geom_before_fullscreen)
            else:
                dialog.geometry("900x700+100+100")
            max_button.config(text="🗖")  # Change icon to maximize
    
    # Make sure dialog stays on top of the parent window
    dialog.transient(parent)
    dialog.grab_set()
    
    # Set styling for a modern look that matches the main application
    style = ttk.Style()
    
    # Configure the style for different elements to match main app
    style.configure('TFrame', background=colors['background'])
    style.configure('Card.TFrame', background=colors['card'])
    
    style.configure('TLabel', 
                   background=colors['background'],
                   foreground=colors['text'],
                   font=('Segoe UI', 10))
    
    style.configure('Header.TLabel', 
                   background=colors['background'],
                   foreground=colors['primary'],
                   font=('Segoe UI', 12, 'bold'))
    
    style.configure('Subheader.TLabel', 
                   background=colors['background'],
                   foreground=colors['text'],
                   font=('Segoe UI', 11, 'bold'))
    
    style.configure('Card.TLabel',
                   background=colors['card'],
                   foreground=colors['text'])
    
    style.configure('TButton', 
                   font=('Segoe UI', 10),
                   background=colors['primary'],
                   foreground='white')
    
    style.map('TButton', 
             background=[('active', colors['accent'])],
             foreground=[('active', 'white')])
    
    style.configure('Primary.TButton', 
                   background=colors['primary'],
                   foreground='white')
    
    style.map('Primary.TButton', 
             background=[('active', '#2980b9')],  # Darker blue
             foreground=[('active', 'white')])
    
    style.configure('Success.TButton', 
                   background=colors['secondary'],
                   foreground='white')
    
    style.map('Success.TButton', 
             background=[('active', '#27ae60')],  # Darker green
             foreground=[('active', 'white')])
    
    style.configure('TNotebook.Tab', 
                   font=('Segoe UI', 10),
                   padding=[12, 8],
                   background=colors['background'],
                   foreground=colors['text'])
    
    style.map('TNotebook.Tab', 
             background=[('selected', colors['primary'])],
             foreground=[('selected', 'white')])
    
    style.configure('TNotebook', 
                   background=colors['background'],
                   tabmargins=[2, 5, 2, 0])
    
    style.configure('TCheckbutton', 
                   background=colors['background'],
                   foreground=colors['text'],
                   font=('Segoe UI', 10))
    
    style.configure('Card.TCheckbutton', 
                   background=colors['card'],
                   foreground=colors['text'],
                   font=('Segoe UI', 10))
    
    style.map('TCheckbutton', 
             background=[('active', colors['background'])],
             foreground=[('active', colors['primary'])])
    
    # Create a custom canvas for the logo/header
    header_frame = ttk.Frame(dialog, style='TFrame')
    header_frame.pack(fill='x', pady=(0, 10))
    
    # Create header with method name and icon
    icon_text = "🧩" if method == "kmeans" else "🔄"
    header_label = ttk.Label(header_frame, 
                           text=f"{icon_text} {method_name} Clustering",
                           style='Header.TLabel')
    header_label.pack(side='left', padx=20, pady=15)
    
    # Create a toolbar
    toolbar = ttk.Frame(header_frame, style='TFrame')
    toolbar.pack(side='right', padx=20, pady=15)
    
    # Add maximize button to toolbar
    max_button = ttk.Button(toolbar, text="🗖", width=3, command=toggle_maximize)
    max_button.pack(side='right', padx=5)
    
    # Create a main container frame that will resize with the window
    main_container = ttk.Frame(dialog, style='TFrame')
    main_container.pack(fill='both', expand=True, padx=20, pady=10)
    
    # Create notebook with tabs
    notebook = ttk.Notebook(main_container)
    notebook.pack(fill='both', expand=True)
    
    # Create tabs with padding for better spacing
    settings_frame = ttk.Frame(notebook, style='TFrame', padding=15)
    preview_frame = ttk.Frame(notebook, style='TFrame', padding=15)
    notebook.add(settings_frame, text="Settings", padding=5)
    notebook.add(preview_frame, text="Preview", padding=5)
    
    # Configure settings frame for responsiveness
    settings_frame.columnconfigure(0, weight=1)
    
    # ---- Settings Tab ----
    # Create a scrollable container for all settings
    settings_canvas = tk.Canvas(settings_frame, bg=colors['background'], 
                              highlightthickness=0)
    settings_canvas.grid(row=0, column=0, sticky='nsew')
    settings_frame.rowconfigure(0, weight=1)
    
    # Add scrollbar
    settings_scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", 
                                     command=settings_canvas.yview)
    settings_scrollbar.grid(row=0, column=1, sticky='ns')
    settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
    
    # Create a frame inside the canvas to hold content
    settings_content = ttk.Frame(settings_canvas, style='TFrame')
    settings_canvas.create_window((0, 0), window=settings_content, anchor="nw", 
                                 tags="settings_content")
    
    # Configure for proper resizing
    def on_settings_configure(event):
        # Update the scrollregion to encompass the inner frame
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        # Set the canvas width to match the settings_frame width
        canvas_width = event.width
        settings_canvas.itemconfig("settings_content", width=canvas_width)
    
    settings_canvas.bind('<Configure>', on_settings_configure)
    settings_content.bind("<Configure>", 
                         lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
    
    # Configure settings_content for grid layout
    settings_content.columnconfigure(0, weight=1)
    
    # Features Card
    features_card = ttk.Frame(settings_content, style='Card.TFrame', padding=15)
    features_card.grid(row=0, column=0, sticky='ew', pady=(0, 15))
    
    # Features title with icon
    ttk.Label(features_card, text="📊 Features for Clustering", 
              style='Subheader.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 10))
    
    # Create scrollable area for features within the card
    features_inner_frame = ttk.Frame(features_card, style='Card.TFrame')
    features_inner_frame.grid(row=1, column=0, sticky='ew')
    features_card.columnconfigure(0, weight=1)
    
    # Organize features in columns for better display
    FEATURES_PER_COLUMN = 8
    
    # Numeric columns for feature selection
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create checkbuttons for features
    feature_vars = {}
    for i, col in enumerate(numeric_cols):
        col_idx = i // FEATURES_PER_COLUMN
        row_idx = i % FEATURES_PER_COLUMN
        
        var = tk.BooleanVar(value=True)
        feature_vars[col] = var
        
        # Create a frame for each checkbutton for better styling
        feature_frame = ttk.Frame(features_inner_frame, style='Card.TFrame')
        feature_frame.grid(row=row_idx, column=col_idx, sticky="w", padx=(col_idx*5, 0), pady=3)
        
        ttk.Checkbutton(feature_frame, text=col, variable=var, 
                      style='Card.TCheckbutton').pack(side='left')
    
    # Parameters Card
    params_card = ttk.Frame(settings_content, style='Card.TFrame', padding=15)
    params_card.grid(row=1, column=0, sticky='ew', pady=(0, 15))
    
    # Parameters title with icon
    ttk.Label(params_card, text="⚙️ Clustering Parameters", 
              style='Subheader.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))
    
    # Configure for grid layout
    params_card.columnconfigure(1, weight=1)
    
    # Create variables for options
    n_clusters_var = tk.IntVar(value=3)
    random_state_var = tk.IntVar(value=42)
    visualize_var = tk.BooleanVar(value=True)
    
    method_specific_vars = {}
    if method == "kmeans":
        method_specific_vars['max_iter'] = tk.IntVar(value=300)
        method_specific_vars['n_init'] = tk.IntVar(value=10)
    elif method == "pam":
        method_specific_vars['max_iter'] = tk.IntVar(value=100)
    elif method == "dbscan":
        method_specific_vars['eps'] = tk.DoubleVar(value=0.5)
        method_specific_vars['min_samples'] = tk.IntVar(value=5)
    elif method == "agnes":
        method_specific_vars['linkage'] = tk.StringVar(value='ward')
        method_specific_vars['affinity'] = tk.StringVar(value='euclidean')
    elif method == "diana":
        method_specific_vars['metric'] = tk.StringVar(value='euclidean')
        method_specific_vars['max_clusters'] = tk.IntVar(value=5)
    
    # Helper function for creating parameter controls
    def create_parameter_row(parent, row, label_text, var, from_val, to_val, tooltip=""):
        # Label
        param_label = ttk.Label(parent, text=label_text, style='Card.TLabel')
        param_label.grid(row=row, column=0, sticky="w", pady=(10, 0))
        
        # Add tooltip if provided
        if tooltip:
            # Create tooltip functionality
            def show_tooltip(event):
                x, y, _, _ = param_label.bbox("insert")
                x += param_label.winfo_rootx() + 25
                y += param_label.winfo_rooty() + 25
                
                # Create a toplevel window
                tip_window = tk.Toplevel(param_label)
                tip_window.wm_overrideredirect(True)
                tip_window.wm_geometry(f"+{x}+{y}")
                
                tip_frame = ttk.Frame(tip_window, style='Card.TFrame', padding=5)
                tip_frame.pack()
                
                # Add a label with the tooltip text
                tip_text = ttk.Label(tip_frame, text=tooltip, 
                                   justify=tk.LEFT, wraplength=300,
                                   style='Card.TLabel')
                tip_text.pack()
                
                # Function to destroy tooltip
                def hide_tooltip():
                    tip_window.destroy()
                
                # Bind events to destroy tooltip
                tip_text.bind("<Leave>", lambda e: hide_tooltip())
                param_label.bind("<Leave>", lambda e: hide_tooltip())
                tip_window.bind("<Leave>", lambda e: hide_tooltip())
            
            # Bind tooltip to label
            param_label.bind("<Enter>", show_tooltip)
        
        # Slider
        slider = ttk.Scale(parent, from_=from_val, to=to_val, variable=var, 
                         orient="horizontal")
        slider.grid(row=row, column=1, sticky="ew", pady=(10, 0), padx=(10, 10))
        
        # Value display with custom styling
        value_frame = ttk.Frame(parent, style='Card.TFrame')
        value_frame.grid(row=row, column=2, sticky="w", pady=(10, 0))
        
        value_display = ttk.Spinbox(value_frame, from_=from_val, to=to_val, 
                                  textvariable=var, width=5)
        value_display.pack(padx=0)
        
        return row + 1
    
    # Helper function for creating dropdown parameters
    def create_dropdown_row(parent, row, label_text, var, values, tooltip=""):
        # Label
        param_label = ttk.Label(parent, text=label_text, style='Card.TLabel')
        param_label.grid(row=row, column=0, sticky="w", pady=(10, 0))
        
        # Add tooltip if provided
        if tooltip:
            # Same tooltip functionality as above
            def show_tooltip(event):
                x, y, _, _ = param_label.bbox("insert")
                x += param_label.winfo_rootx() + 25
                y += param_label.winfo_rooty() + 25
                
                tip_window = tk.Toplevel(param_label)
                tip_window.wm_overrideredirect(True)
                tip_window.wm_geometry(f"+{x}+{y}")
                
                tip_frame = ttk.Frame(tip_window, style='Card.TFrame', padding=5)
                tip_frame.pack()
                
                tip_text = ttk.Label(tip_frame, text=tooltip, 
                                   justify=tk.LEFT, wraplength=300,
                                   style='Card.TLabel')
                tip_text.pack()
                
                def hide_tooltip():
                    tip_window.destroy()
                
                tip_text.bind("<Leave>", lambda e: hide_tooltip())
                param_label.bind("<Leave>", lambda e: hide_tooltip())
                tip_window.bind("<Leave>", lambda e: hide_tooltip())
            
            param_label.bind("<Enter>", show_tooltip)
        
        # Combobox for dropdown
        dropdown = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=15)
        dropdown.grid(row=row, column=1, sticky="w", pady=(10, 0), padx=(10, 10))
        
        return row + 1
    
    # Add parameters with tooltips
    row_idx = 1
    
    # Number of clusters
    if method in ["kmeans", "pam", "agnes", "diana"]:
        row_idx = create_parameter_row(
            params_card, row_idx, "Number of clusters:", n_clusters_var, 2, 10,
            "The number of clusters to form as well as the number of centroids to generate."
        )
    
    # Method-specific options
    if method == "kmeans":
        row_idx = create_parameter_row(
            params_card, row_idx, "Max iterations:", method_specific_vars['max_iter'], 100, 1000,
            "Maximum number of iterations of the k-means algorithm for a single run."
        )
        row_idx = create_parameter_row(
            params_card, row_idx, "Number of initializations:", method_specific_vars['n_init'], 1, 20,
            "Number of time the k-means algorithm will be run with different centroid seeds."
        )
    elif method == "pam":
        row_idx = create_parameter_row(
            params_card, row_idx, "Max iterations:", method_specific_vars['max_iter'], 50, 500,
            "Maximum number of iterations of the PAM algorithm."
        )
    elif method == "dbscan":
        row_idx = create_parameter_row(
            params_card, row_idx, "Epsilon (eps):", method_specific_vars['eps'], 0.1, 2.0,
            "The maximum distance between two samples for them to be considered as in the same neighborhood."
        )
        row_idx = create_parameter_row(
            params_card, row_idx, "Min samples:", method_specific_vars['min_samples'], 1, 20,
            "The number of samples in a neighborhood for a point to be considered as a core point."
        )
    elif method == "agnes":
        row_idx = create_dropdown_row(
            params_card, row_idx, "Linkage:", method_specific_vars['linkage'], 
            ['ward', 'complete', 'average', 'single'],
            "Linkage criterion to use: 'ward' minimizes variance, 'complete' uses maximum distances, 'average' uses average distances, 'single' uses minimum distances."
        )
        row_idx = create_dropdown_row(
            params_card, row_idx, "Affinity:", method_specific_vars['affinity'], 
            ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
            "Metric used to compute the linkage. Note that 'ward' linkage only supports 'euclidean' affinity."
        )
    
    # Random state
    if method in ['kmeans', 'pam', 'dbscan']:
        row_idx = create_parameter_row(
            params_card, row_idx, "Random state:", random_state_var, 0, 100,
            "Determines random number generation for centroid initialization."
        )
    
    # Visualization Card
    viz_card = ttk.Frame(settings_content, style='Card.TFrame', padding=15)
    viz_card.grid(row=2, column=0, sticky='ew', pady=(0, 15))
    
    # Visualization title with icon
    ttk.Label(viz_card, text="🎨 Visualization Options", 
              style='Subheader.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 10))
    
    # Visualization checkbox
    viz_check = ttk.Checkbutton(viz_card, text="Visualize clustering results", 
                              variable=visualize_var, style='Card.TCheckbutton')
    viz_check.grid(row=1, column=0, sticky="w", pady=(5, 5))
    
  
    # Button Card
    button_card = ttk.Frame(settings_content, style='Card.TFrame', padding=15)
    button_card.grid(row=3, column=0, sticky='ew', pady=(0, 15))
    
    # Create a container for buttons
    button_container = ttk.Frame(button_card, style='Card.TFrame')
    button_container.pack(side='right')
    
    # Result variable for storing the clustering result
    result = [None]
    
    # Function to run clustering with the selected parameters
    def run_clustering():
        # Get selected features
        selected_features = [col for col, var in feature_vars.items() if var.get()]
        
        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature for clustering")
            return
        
        try:
            # Filter data to selected features
            filtered_data = data[selected_features].copy()
            
            # Show progress indicator
            progress_window = tk.Toplevel(dialog)
            progress_window.title("Processing")
            progress_window.geometry("350x150")
            progress_window.resizable(False, False)
            progress_window.transient(dialog)
            progress_window.grab_set()
            progress_window.configure(bg=colors['background'])
            
            # Center progress window on dialog
            progress_window.update_idletasks()
            dialog_x = dialog.winfo_rootx()
            dialog_y = dialog.winfo_rooty()
            dialog_width = dialog.winfo_width()
            dialog_height = dialog.winfo_height()
            progress_width = progress_window.winfo_width()
            progress_height = progress_window.winfo_height()
            
            x = dialog_x + (dialog_width // 2) - (progress_width // 2)
            y = dialog_y + (dialog_height // 2) - (progress_height // 2)
            progress_window.geometry(f"+{x}+{y}")
            
            # Progress window contents
            progress_frame = ttk.Frame(progress_window, style='Card.TFrame', padding=15)
            progress_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            ttk.Label(progress_frame, text="Processing clusters...", 
                    style='Subheader.TLabel').pack(pady=(0, 15))
            
            progress = ttk.Progressbar(progress_frame, mode='indeterminate', length=300)
            progress.pack(fill='x')
            progress.start(10)  # Faster animation
            
            # Update UI
            dialog.update_idletasks()
            
            # Perform clustering
            if method == "kmeans":
                clustered_data, metrics = perform_kmeans_clustering(
                    filtered_data,
                    n_clusters=n_clusters_var.get(),
                    random_state=random_state_var.get(),
                    max_iter=method_specific_vars['max_iter'].get(),
                    n_init=method_specific_vars['n_init'].get(),
                    visualize=visualize_var.get()
                )
            elif method == "pam":  # PAM
                clustered_data, metrics = perform_pam_clustering(
                    filtered_data,
                    n_clusters=n_clusters_var.get(),
                    random_state=random_state_var.get(),
                    max_iter=method_specific_vars['max_iter'].get(),
                    visualize=visualize_var.get()
                )
            
            elif method == "dbscan":
                clustered_data, metrics = perform_dbscan_clustering(
                    filtered_data,
                    eps=method_specific_vars['eps'].get(),
                    min_samples=method_specific_vars['min_samples'].get(),
                    visualize=visualize_var.get()
                )
            
            elif method == "agnes":
                clustered_data, metrics = perform_agnes_clustering(
                    filtered_data,
                    n_clusters=n_clusters_var.get(),
                    linkage=method_specific_vars['linkage'].get(),
                    affinity=method_specific_vars['affinity'].get(),
                    visualize=visualize_var.get()
                )

            elif method == "diana":
                clustered_data, metrics = perform_diana_clustering(
                    filtered_data,
                    n_clusters=n_clusters_var.get(),
                    visualize=visualize_var.get()
                )
            
            # Add the cluster column to the original data
            result_data = data.copy()
            result_data['cluster'] = clustered_data['cluster']
            
            # Close progress window
            progress_window.destroy()
            
            # For AGNES, offer to show dendrogram
            if method in ['agnes', 'diana']:
                import tkinter.messagebox as messagebox
                if messagebox.askyesno("Show Dendrogram", "Would you like to see the hierarchical clustering dendrogram?"):
                    show_dendrogram(
                        filtered_data, 
                        linkage=method_specific_vars['linkage'].get(),
                        affinity=method_specific_vars['affinity'].get()
                    )
            
            # Store result and close dialog
            result[0] = (result_data, n_clusters_var.get(), metrics)
            dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            if 'progress_window' in locals():
                progress_window.destroy()
    
    # Add a help button
    def show_help():
        help_window = tk.Toplevel(dialog)
        help_window.title(f"{method_name} Clustering Help")
        help_window.geometry("550x450")
        help_window.transient(dialog)
        help_window.grab_set()
        help_window.configure(bg=colors['background'])
        
        # Create help content frame
        help_frame = ttk.Frame(help_window, style='Card.TFrame', padding=20)
        help_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Help title
        ttk.Label(help_frame, text=f"{icon_text} {method_name} Clustering Help", 
                style='Header.TLabel').pack(anchor='w', pady=(0, 15))
        
        # Help content in a scrollable text area
        help_text_frame = ttk.Frame(help_frame, style='Card.TFrame')
        help_text_frame.pack(fill='both', expand=True)
        
        help_canvas = tk.Canvas(help_text_frame, background=colors['card'], highlightthickness=0)
        help_canvas.pack(side='left', fill='both', expand=True)
        
        help_scrollbar = ttk.Scrollbar(help_text_frame, orient="vertical", command=help_canvas.yview)
        help_scrollbar.pack(side='right', fill='y')
        
        help_canvas.configure(yscrollcommand=help_scrollbar.set)
        
        help_content_frame = ttk.Frame(help_canvas, style='Card.TFrame')
        help_canvas.create_window((0, 0), window=help_content_frame, anchor="nw")
        
        help_content_frame.bind("<Configure>", 
                              lambda e: help_canvas.configure(scrollregion=help_canvas.bbox("all")))
        
        if method == "kmeans":
            help_sections = [
                ("What is K-means?", 
                "K-means clustering partitions data into k clusters where each observation belongs to "
                "the cluster with the nearest mean (cluster centroid)."),

                ("Parameters Explained", """
        • Number of clusters: The number of clusters to form.
        • Max iterations: Maximum number of iterations for the algorithm to converge.
        • Number of initializations: How many times to run the algorithm with different initial centroids.
        • Random state: For reproducible results.
                """),

                ("How It Works", """
        1. The algorithm starts by initializing k points as cluster centers (centroids).
        2. It then assigns each data point to the nearest centroid.
        3. The centroids are recalculated as the mean of all points assigned to that cluster.
        4. Steps 2 and 3 are repeated until convergence or reaching max iterations.
                """),

                ("Tips", """
        • Try different numbers of clusters to find the optimal grouping.
        • Higher Silhouette Score indicates better-defined clusters (range -1 to 1).
        • Lower Intra-cluster distances indicate more compact clusters.
        • Higher Inter-cluster distances indicate better separated clusters.
        • K-means works best with spherical clusters of similar size.
                """)
            ]
        elif method == "pam":
            help_sections = [
                ("What is PAM?", 
                "PAM (Partitioning Around Medoids) clustering is similar to K-means but uses actual data points "
                "(medoids) as the center of clusters, making it more robust to outliers."),

                ("Parameters Explained", """
        • Number of clusters: The number of clusters to form.
        • Max iterations: Maximum number of iterations for the algorithm to converge.
        • Random state: For reproducible results.
                """),

                ("How It Works", """
        1. The algorithm starts by randomly selecting k data points as medoids.
        2. It then assigns each data point to the nearest medoid.
        3. For each cluster, the algorithm finds the point that minimizes the sum of distances to other points in the cluster.
        4. If a better medoid is found, it replaces the current one.
        5. Steps 2-4 are repeated until no change or reaching max iterations.
                """),

                ("Tips", """
        • PAM is less sensitive to outliers than K-means.
        • Works well with smaller datasets.
        • Try different numbers of clusters to find the optimal grouping.
        • Higher Silhouette Score indicates better-defined clusters (range -1 to 1).
        • Lower Intra-cluster distances indicate more compact clusters.
        • Higher Inter-cluster distances indicate better separated clusters.
                """)
            ]
        elif method == "dbscan":
            help_sections = [
                ("What is DBSCAN?", 
                "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. "
                "It groups points that are closely packed together, marking outliers as noise."),

                ("Parameters Explained", """
        • Epsilon (eps): The maximum distance between two samples for them to be considered as in the same neighborhood.
        • Min samples: The number of samples in a neighborhood for a point to be considered as a core point.
        • Random state: For reproducible results.
                """),

                ("How It Works", """
        1. The algorithm starts by selecting a random point and finding all points within epsilon (eps) distance.
        2. If the number of points in the neighborhood exceeds the min_samples threshold, a cluster is formed.
        3. Points that are reachable from other points are assigned to the cluster; others are considered noise.
        4. Steps 1-3 continue until all points are processed.
                """),

                ("Tips", """
        • DBSCAN works well for datasets with noise and clusters of arbitrary shape.
        • Choosing the right eps and min_samples is crucial for good clustering.
        • You can use the silhouette score or other metrics to evaluate the clustering quality.
        • DBSCAN does not require specifying the number of clusters upfront, making it useful for data with unknown cluster counts.
                """)
            ]
        elif method == "agnes":
            help_sections = [
                ("What is AGNES?", 
                "AGNES (Agglomerative Nesting or Hierarchical Clustering) builds a hierarchy of clusters starting with each data point as its own cluster and progressively merging clusters."),

                ("Parameters Explained", """
        • Number of clusters: The final number of clusters to form.
        • Linkage: Determines which distance metric to use between sets of observations:
          - Ward: Minimizes the variance of clusters being merged (default)
          - Complete: Uses the maximum distances between all observations of two clusters
          - Average: Uses the average distances between all observations of two clusters
          - Single: Uses the minimum distances between all observations of two clusters
        • Affinity: The metric used to compute the linkage (euclidean, manhattan, cosine, etc.)
        • Random state: For reproducible results.
                """),

                ("How It Works", """
        1. The algorithm starts by assigning each data point to its own cluster.
        2. It then iteratively merges the two most similar clusters based on the chosen linkage criterion.
        3. This process continues until the specified number of clusters is reached.
        4. The result is a dendrogram-like structure showing the hierarchical clustering.
                """),

                ("Tips", """
        • AGNES can identify hierarchical structures in data that other methods might miss.
        • Ward linkage tends to create more balanced, equal-sized clusters.
        • Different linkage criteria create different cluster shapes: try several to find the best for your data.
        • Higher Silhouette Score indicates better-defined clusters (range -1 to 1).
        • If using ward linkage, you must use euclidean affinity.
        • Hierarchical clustering can be interpreted at different levels of the hierarchy.
                """)
            ]
        elif method == "diana":
            help_sections = [
                ("What is DIANA?", 
                "DIANA (Divisive Analysis Clustering) is a hierarchical clustering algorithm that starts with all data in a single cluster and recursively splits it into smaller clusters."),

                ("Parameters Explained", """
        • Number of clusters: The final number of clusters to form.
        • Dissimilarity metric: Determines how dissimilar two observations are (typically Euclidean distance).
        • Splitting criterion: Determines how the algorithm decides which cluster to split and how (e.g., maximum diameter or average dissimilarity).
                """),

                ("How It Works", """
        1. DIANA begins with all data points in one cluster.
        2. It identifies the cluster with the largest internal dissimilarity (diameter).
        3. From that cluster, it selects the most dissimilar point and creates a new cluster.
        4. It then assigns points to the new or original cluster based on their dissimilarities.
        5. This process is repeated until the specified number of clusters is formed.
                """),

                ("Tips", """
        • DIANA is especially good at finding large, distinct clusters first.
        • Unlike AGNES, which builds up from individual points, DIANA breaks down from the whole.
        • Works best on datasets with clear separation between groups.
        • It's slower than AGNES for large datasets due to recursive splitting.
        • Results are deterministic and do not require a random seed.
                """)
            ]
        
        for i, (title, content) in enumerate(help_sections):
            section_frame = ttk.Frame(help_content_frame, style='Card.TFrame')
            section_frame.pack(fill='x', pady=(0, 15), padx=5)
            
            ttk.Label(section_frame, text=title, 
                    style='Subheader.TLabel').pack(anchor='w', pady=(0, 5))
            
            text_widget = tk.Text(section_frame, wrap=tk.WORD, height=content.count('\n')+3,
                                width=50, font=('Segoe UI', 10),
                                background=colors['card'], foreground=colors['text'],
                                borderwidth=0, highlightthickness=0)
            text_widget.pack(fill='x')
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Adjust scrollable region
        help_content_frame.update_idletasks()
        help_canvas.configure(scrollregion=help_canvas.bbox("all"))
        
        # Close button
        ttk.Button(help_frame, text="Close", style='Primary.TButton',
                 command=help_window.destroy).pack(pady=(15, 0))
    
    # Add buttons with icons
    ttk.Button(button_container, text="Run Clustering ▶", style='Success.TButton',
             command=run_clustering).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_container, text="❓ Help", style='Primary.TButton',
             command=show_help).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_container, text="Cancel ✖", 
             command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    # ---- Preview Tab ----
    # Create a card for the preview
    preview_card = ttk.Frame(preview_frame, style='Card.TFrame', padding=15)
    preview_card.pack(fill='both', expand=True)
    
    preview_header = ttk.Label(preview_card, 
                             text="Clustering Visualization Preview", 
                             style='Subheader.TLabel')
    preview_header.pack(pady=(0, 15))
    
    preview_content = ttk.Label(preview_card, 
                              text="After clustering, a visualization will appear here.",
                              style='Card.TLabel')
    preview_content.pack(fill='both', expand=True)
    
    # Add a placeholder image for visual appeal
    placeholder_frame = ttk.Frame(preview_card, style='Card.TFrame', height=300)
    placeholder_frame.pack(fill='x', pady=10)
    
    # Create a simple visualization placeholder
    canvas = tk.Canvas(placeholder_frame, bg=colors['card'], height=300, 
                      highlightthickness=0)
    canvas.pack(fill='x')
    
    # Draw some placeholder cluster-like shapes
    def draw_cluster_placeholder():
        canvas.delete("all")  # Clear canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1:  # Canvas not yet properly sized
            canvas.after(100, draw_cluster_placeholder)
            return
        
        # Draw background grid
        for i in range(0, width, 30):
            canvas.create_line(i, 0, i, height, fill="#f0f0f0")
        for i in range(0, height, 30):
            canvas.create_line(0, i, width, i, fill="#f0f0f0")
        
        # Draw axes
        canvas.create_line(50, height-50, width-50, height-50, 
                          fill="#555555", width=2, arrow=tk.LAST)
        canvas.create_line(50, height-50, 50, 50, 
                          fill="#555555", width=2, arrow=tk.LAST)
        
        # X and Y labels
        canvas.create_text(width-40, height-40, text="X", fill="#555555", font=('Segoe UI', 9))
        canvas.create_text(40, 40, text="Y", fill="#555555", font=('Segoe UI', 9))
        
        # Draw some clusters with different colors
        centers = [
            (width * 0.3, height * 0.4, colors['primary']),
            (width * 0.7, height * 0.3, colors['secondary']),
            (width * 0.5, height * 0.7, colors['accent'])
        ]
        
        # Draw points for each cluster
        for cx, cy, color in centers:
            # Draw center point
            canvas.create_oval(cx-8, cy-8, cx+8, cy+8, 
                              fill=color, outline="white", width=2)
            
            # Draw cluster points
            for i in range(12):
                angle = random.random() * 2 * 3.14159
                dist = random.random() * 60 + 20
                x = cx + math.cos(angle) * dist
                y = cy + math.sin(angle) * dist
                canvas.create_oval(x-4, y-4, x+4, y+4, fill=color, outline="")
    
    # Need to import these for the placeholder
    import random
    import math
    
    # Schedule the drawing after the canvas is properly sized
    canvas.after(100, draw_cluster_placeholder)
    
    # Wait for the dialog to close
    parent.wait_window(dialog)
    
    return result[0]
