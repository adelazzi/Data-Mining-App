import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk  # For modern themes
import pandas as pd
import numpy as np
from visualizations.box_plot import create_box_plot
from visualizations.qq_plot import create_qq_plot
from visualizations.scatter_plot import create_scatter_plot
from visualizations.cluster_plot import create_kmeans_plot, create_pam_plot, create_dbscan_plot, create_agnes_plot
from data_loader import load_dataset
from data_preprocessor import (
    clean_data, normalize_data, handle_missing_values, 
    show_preprocessing_dialog, show_progress_bar
)
from classification import (
    train_knn_model, train_naive_bayes_model, train_decision_tree_model,
    train_linear_regression_model, train_neural_network_model,
    show_classification_dialog, predict_new_value
)
from clustering import (
    perform_kmeans_clustering, 
    perform_pam_clustering,
    perform_dbscan_clustering,
    perform_agnes_clustering,
    show_clustering_dialog
)

class DataVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Visualization & Analysis Tool")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Apply modern theme and custom styling
        self.setup_styles()
        
        # Initialize variables
        self.current_data = None
        self.filename = None
        self.current_model = None
        self.model_type = None
        self.target_column = None
        self.feature_columns = None
        
        # Create main container that will handle resizing
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Configure rows and columns with weights for responsiveness
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=0)  # Header
        self.main_container.rowconfigure(1, weight=1)  # Content
        self.main_container.rowconfigure(2, weight=0)  # Status bar
        
        # Setup UI components
        self.create_header()
        self.setup_menu()
        self.create_content_area()
        self.create_status_bar()

    def setup_styles(self):
        """Setup custom styles for the application with a cleaner, more professional look"""
        self.style = ttk.Style(self.root)
        
        # Use a clean, modern theme as base
        self.style.theme_use("clam")  # More neutral base theme
        
        # Configure fonts - use a single font family for consistency
        default_font = ('Segoe UI', 10)
        header_font = ('Segoe UI', 11, 'bold')
        title_font = ('Segoe UI', 12, 'bold')
        
        # Configure colors - simplified neutral palette
        bg_color = '#f8f8f8'         # Light gray background
        accent_color = '#505050'     # Dark gray for accents
        text_color = '#333333'       # Dark gray for text
        highlight_color = '#4a4a4a'  # Slightly darker for highlights
        
        # Apply colors to root window
        self.root.configure(background=bg_color)
        
        # Custom styles for various widgets - more consistent and clean
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=text_color, font=default_font)
        self.style.configure('TButton', font=default_font, padding=6)
        
        # Header styles - more subtle
        self.style.configure('Header.TFrame', background=accent_color)
        self.style.configure('Header.TLabel', background=accent_color, foreground='white', font=header_font)
        self.style.configure('Title.TLabel', font=title_font, background=bg_color, foreground=text_color)
        
        # Status bar style - subtle
        self.style.configure('Status.TLabel', background='#eeeeee', foreground=text_color, font=('Segoe UI', 9))
        
        # File info style - subtle highlight
        self.style.configure('FileInfo.TFrame', background='#f0f0f0', relief='flat')
        self.style.configure('FileInfo.TLabel', background='#f0f0f0', foreground=text_color)
        
        # Treeview styling - cleaner look
        self.style.configure('Treeview', 
                             background='white', 
                             foreground=text_color, 
                             rowheight=25, 
                             fieldbackground='white',
                             font=default_font,
                             borderwidth=1)
        self.style.configure('Treeview.Heading', 
                             font=('Segoe UI', 10, 'bold'),
                             background='#e8e8e8',
                             relief='flat')
        
        # Map styles for button hover effects - subtle
        self.style.map('TButton', 
                       background=[('active', '#e0e0e0'), ('pressed', '#d0d0d0')],
                       foreground=[('active', text_color), ('pressed', text_color)])
        
        # Treeview selection color - subtle
        self.style.map('Treeview', 
                      background=[('selected', '#d0d0d0')],
                      foreground=[('selected', text_color)])

    def create_header(self):
        """Create a clean, minimal header for the application"""
        # Header container
        self.header_frame = ttk.Frame(self.main_container, style='Header.TFrame')
        self.header_frame.grid(row=0, column=0, sticky='ew')
        
        # App title and logo container
        title_container = ttk.Frame(self.header_frame, style='Header.TFrame')
        title_container.pack(fill='x', padx=15, pady=10)
        
        # Simplified icon
        logo_text = "Data Visualization"
        logo_label = ttk.Label(title_container, text=logo_text, style='Header.TLabel')
        logo_label.pack(side='left')
        
        # Version info - more subtle
        version_label = ttk.Label(title_container, text="v1.0", style='Header.TLabel', font=('Segoe UI', 8))
        version_label.pack(side='right')

    def setup_menu(self):
        """Setup the menu bar and its options."""
        self.menu_bar = tk.Menu(self.root)
        self.setup_file_menu()
        self.setup_preprocess_menu()
        self.setup_classification_menu()
        self.setup_regression_menu()
        self.setup_clustering_menu()
        self.setup_visualize_menu()
        self.root.config(menu=self.menu_bar)

    def setup_file_menu(self):
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open...", command=self.open_file)
        self.file_menu.add_command(label="Edit", command=self.edit_data)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

    def setup_preprocess_menu(self):
        self.preprocess_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.preprocess_menu.add_command(label="Clean Data", command=self.clean_data)
        self.preprocess_menu.add_command(label="Normalize Data", command=self.normalize_data)
        self.preprocess_menu.add_command(label="Handle Missing Values", command=self.handle_missing_values)
        self.preprocess_menu.add_command(label="Advanced Preprocessing...", command=self.show_preprocessing_dialog)
        self.menu_bar.add_cascade(label="Preprocess", menu=self.preprocess_menu)

    def setup_classification_menu(self):
        self.classification_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.classification_menu.add_command(label="Train KNN Model", command=self.train_knn)
        self.classification_menu.add_command(label="Train Naive Bayes Model", command=self.train_naive_bayes)
        self.classification_menu.add_command(label="Train Decision Tree Model", command=self.train_decision_tree)
        self.classification_menu.add_separator()
        self.classification_menu.add_command(label="Classify New Value", command=self.classify_new_value)
        self.menu_bar.add_cascade(label="Classification", menu=self.classification_menu)

    def setup_regression_menu(self):
        self.regression_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.regression_menu.add_command(label="Train Linear Regression Model", command=self.train_linear_regression)
        self.regression_menu.add_command(label="Train Neural Network Model", command=self.train_neural_network)
        self.regression_menu.add_separator()
        self.menu_bar.add_cascade(label="Regression", menu=self.regression_menu)

    def setup_clustering_menu(self):
        self.clustering_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.clustering_menu.add_command(label="K-Means Clustering", command=self.perform_kmeans_clustering)
        self.clustering_menu.add_command(label="PAM Clustering", command=self.perform_pam_clustering)
        self.clustering_menu.add_command(label="DBSCAN Clustering", command=self.perform_dbscan_clustering)
        self.clustering_menu.add_command(label="AGNES Clustering", command=self.perform_agnes_clustering)
        self.clustering_menu.add_command(label="DIANA Clustering", command=self.perform_diana_clustering)
        self.menu_bar.add_cascade(label="Clustering", menu=self.clustering_menu)

    def setup_visualize_menu(self):
        self.visualize_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.visualize_menu.add_command(label="Box Plot", command=self.show_box_plot)
        self.visualize_menu.add_command(label="QQ Plot", command=self.show_qq_plot)
        self.visualize_menu.add_command(label="Scatter Plot", command=self.show_scatter_plot)
        self.menu_bar.add_cascade(label="Visualize", menu=self.visualize_menu)

    def create_content_area(self):
        """Create the main content area with responsive layout"""
        # Content container with grid for better organization
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.grid(row=1, column=0, sticky='nsew', padx=15, pady=10)
        
        # Configure responsive grid
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=0)  # File info
        self.content_frame.rowconfigure(1, weight=1)  # Data view
        
        # Create file info section with improved visuals
        self.create_file_info_section()
        
        # Create data view with responsive design
        self.create_data_view()

    def create_file_info_section(self):
        """Create a cleaner file information section"""
        self.file_info_frame = ttk.Frame(self.content_frame, style='FileInfo.TFrame', padding=(10, 8))
        self.file_info_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        # File label in a single container for better alignment
        file_label_container = ttk.Frame(self.file_info_frame, style='FileInfo.TFrame')
        file_label_container.pack(side='left', fill='x', expand=True)
        
        # Simpler file label
        self.file_label = ttk.Label(file_label_container, text="No file opened", style='FileInfo.TLabel')
        self.file_label.pack(side='left', fill='x')
        
        # Add quick action buttons on the right - simpler design
        self.quick_actions_frame = ttk.Frame(self.file_info_frame, style='FileInfo.TFrame')
        self.quick_actions_frame.pack(side='right')
        
        # Open file button - more subtle
        self.open_button = ttk.Button(self.quick_actions_frame, text="Open File", command=self.open_file)
        self.open_button.pack(side='left', padx=5)
        
        # Edit data button
        self.edit_button = ttk.Button(self.quick_actions_frame, text="Edit Data", command=self.edit_data)
        self.edit_button.pack(side='left', padx=5)

    def create_data_view(self):
        """Create a clean, minimal data view"""
        # Container for data view with subtle border
        self.data_view_frame = ttk.Frame(self.content_frame, padding=1, relief="solid", borderwidth=1)
        self.data_view_frame.grid(row=1, column=0, sticky='nsew')
        
        # Configure the data view to be responsive
        self.data_view_frame.columnconfigure(0, weight=1)
        self.data_view_frame.rowconfigure(0, weight=0)  # Data info bar
        self.data_view_frame.rowconfigure(1, weight=1)  # Treeview
        
        # Data info bar - cleaner look
        self.data_info_frame = ttk.Frame(self.data_view_frame, padding=5)
        self.data_info_frame.grid(row=0, column=0, sticky='ew')
        
        # Labels to show dataset statistics - simplified
        stats_frame = ttk.Frame(self.data_info_frame)
        stats_frame.pack(side='left')
        
        self.rows_label = ttk.Label(stats_frame, text="Rows: 0")
        self.rows_label.pack(side='left', padx=10)
        
        ttk.Separator(stats_frame, orient='vertical').pack(side='left', fill='y', padx=5, pady=2)
        
        self.columns_label = ttk.Label(stats_frame, text="Columns: 0")
        self.columns_label.pack(side='left', padx=10)
        
        # Subtle separator
        ttk.Separator(self.data_info_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Create improved table with container for table and scrollbars
        self.table_container = ttk.Frame(self.data_view_frame)
        self.table_container.grid(row=1, column=0, sticky='nsew')
        
        # Configure the table container to be responsive
        self.table_container.columnconfigure(0, weight=1)
        self.table_container.rowconfigure(0, weight=1)
        
        # Create Treeview with styled appearance
        self.tree = ttk.Treeview(self.table_container)
        self.tree.grid(row=0, column=0, sticky='nsew')
        
        # Add vertical scrollbar
        vscrollbar = ttk.Scrollbar(self.table_container, orient="vertical", command=self.tree.yview)
        vscrollbar.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=vscrollbar.set)
        
        # Add horizontal scrollbar
        hscrollbar = ttk.Scrollbar(self.table_container, orient="horizontal", command=self.tree.xview)
        hscrollbar.grid(row=1, column=0, sticky='ew')
        self.tree.configure(xscrollcommand=hscrollbar.set)

    def create_status_bar(self):
        """Create a clean, minimal status bar"""
        status_frame = ttk.Frame(self.main_container)
        status_frame.grid(row=2, column=0, sticky='ew')
        
        # Subtle separator above status bar
        separator = ttk.Separator(status_frame, orient='horizontal')
        separator.pack(fill='x')
        
        # Status bar - cleaner look
        status_content = ttk.Frame(status_frame, style='Status.TLabel')
        status_content.pack(fill='x')
        
        self.status_bar = ttk.Label(
            status_content, 
            text="Ready", 
            anchor="w", 
            padding=(10, 5),
            style='Status.TLabel'
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Simplified model indicator
        self.model_indicator = ttk.Label(
            status_content,
            text="No Model",
            padding=(5, 5),
            style='Status.TLabel'
        )
        self.model_indicator.pack(side=tk.RIGHT, padx=10)

    def update_status(self, message):
        """Update the status bar with a message.""" 
        self.status_bar.config(text=message)
        
        # Also update model indicator if a model is trained
        if self.current_model is not None:
            model_name = self.model_type.replace('_', ' ').title() if self.model_type else "Unknown"
            self.model_indicator.config(text=f"Model: {model_name}")
        else:
            self.model_indicator.config(text="No Model")

    def update_data_view(self):
        """Update the data view with the current dataset.""" 
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.current_data is None:
            # Reset data info labels
            self.rows_label.config(text="Rows: 0")
            self.columns_label.config(text="Columns: 0")
            return
        
        # Update data info labels
        self.rows_label.config(text=f"Rows: {len(self.current_data)}")
        self.columns_label.config(text=f"Columns: {len(self.current_data.columns)}")
        
        # Configure columns with responsive widths
        self.tree["columns"] = list(self.current_data.columns)
        self.tree["show"] = "headings"
        
        # Calculate column widths based on content
        for col in self.current_data.columns:
            # Convert column data to strings and get the maximum length
            max_width = max(
                len(str(col)),  # Header length
                *[len(str(val)) for val in self.current_data[col].head(100)]  # Sample data length
            )
            
            # Set a reasonable width (characters * average width)
            width = min(max_width * 10, 300)  # Cap at 300 pixels
            
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, minwidth=50)
        
        # Insert data rows with alternating colors
        for i, row in self.current_data.iterrows():
            values = list(row)
            
            # Convert any non-string values to strings for display
            values = [str(val) if not isinstance(val, str) else val for val in values]
            
            # Add row to treeview
            self.tree.insert("", tk.END, values=values, tags=('odd' if i % 2 else 'even',))
        
        # Configure row tags for alternating colors - more subtle
        self.tree.tag_configure('odd', background='#f5f5f5')
        self.tree.tag_configure('even', background='white')

    def open_file(self):
        """Open a data file and load its content with improved UI feedback.""" 
        filetypes = [
            ("All supported files", "*.arff *.csv *.xlsx *.xls"),
            ("ARFF files", "*.arff"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
        
        # Show file dialog
        self.filename = filedialog.askopenfilename(title="Open Data File", filetypes=filetypes)
        if not self.filename:
            return
            
        try:
            # Show loading progress
            progress_window, progress_bar, progress_label = show_progress_bar(
                self.root, 
                "Loading File",
                max_value=100
            )
            
            progress_label.config(text="Reading file...")
            progress_bar["value"] = 30
            progress_window.update()
            
            # Load the dataset
            self.current_data = load_dataset(self.filename)
            
            progress_bar["value"] = 70
            progress_label.config(text="Preparing data view...")
            progress_window.update()
            
            # Update the interface
            self.update_data_view()
            
            # Update file info - simpler display
            filename_display = self.filename
            if len(filename_display) > 60:
                filename_display = "..." + filename_display[-57:]
                
            self.file_label.config(text=f"File: {filename_display}")
            
            progress_bar["value"] = 100
            progress_window.destroy()
            
            self.update_status(f"File loaded successfully: {self.filename.split('/')[-1]}")
            
        except Exception as e:
            if 'progress_window' in locals() and progress_window:
                progress_window.destroy()
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.update_status("Failed to load file.")

    def edit_data(self):
        """Edit the current data in a simple editor"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        # Create a simple editor window
        editor = tk.Toplevel(self.root)
        editor.title("Edit Data")
        editor.geometry("800x600")
        
        # TODO: Implement a proper data editor
        # For now, just show a message
        ttk.Label(editor, text="Data editing functionality will be implemented in future versions.",
                  padding=20).pack(expand=True)
        
        self.update_status("Data editor opened.")
    
    def clean_data(self):
        """Clean the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            self.current_data = clean_data(self.current_data)
            self.update_data_view()
            self.update_status("Data cleaned successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clean data: {str(e)}")
    
    def normalize_data(self):
        """Normalize the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            self.current_data = normalize_data(self.current_data)
            self.update_data_view()
            self.update_status("Data normalized successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to normalize data: {str(e)}")
    
    def handle_missing_values(self):
        """Handle missing values in the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            self.current_data = handle_missing_values(self.current_data)
            self.update_data_view()
            self.update_status("Missing values handled successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle missing values: {str(e)}")
    
    def show_preprocessing_dialog(self):
        """Show the advanced preprocessing dialog"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_preprocessing_dialog(self.root, self.current_data)
            if result is not None:
                self.current_data = result
                self.update_data_view()
                self.update_status("Data preprocessing completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing error: {str(e)}")
    
    def train_knn(self):
        """Train a KNN model on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_classification_dialog(self.root, self.current_data, "knn")
            if result is not None:
                self.current_model, self.model_type, self.target_column, self.feature_columns = result
                self.update_status("KNN model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train KNN model: {str(e)}")
    
    def train_naive_bayes(self):
        """Train a Naive Bayes model on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_classification_dialog(self.root, self.current_data, "naive_bayes")
            if result is not None:
                self.current_model, self.model_type, self.target_column, self.feature_columns = result
                self.update_status("Naive Bayes model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train Naive Bayes model: {str(e)}")
    
    def train_decision_tree(self):
        """Train a Decision Tree model on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_classification_dialog(self.root, self.current_data, "decision_tree")
            if result is not None:
                self.current_model, self.model_type, self.target_column, self.feature_columns = result
                self.update_status("Decision Tree model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train Decision Tree model: {str(e)}")
    
    def classify_new_value(self):
        """Classify a new value using the trained model"""
        if self.current_model is None:
            messagebox.showinfo("No Model", "Please train a classification model first.")
            return
        
        if self.model_type not in ["knn", "naive_bayes", "decision_tree"]:
            messagebox.showinfo("Wrong Model Type", "This function requires a classification model.")
            return
            
        try:
            # Use the predict_new_value function from classification.py
            predict_new_value(self.root, self.current_model, self.target_column, 
                             self.feature_columns, self.model_type)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify new value: {str(e)}")
    

    def train_linear_regression(self):
        """Train a Linear Regression model on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_classification_dialog(self.root, self.current_data, "linear_regression")
            if result is not None:
                self.current_model, self.model_type, self.target_column, self.feature_columns = result
                self.update_status("Linear Regression model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train Linear Regression model: {str(e)}")
    
    def train_neural_network(self):
        """Train a Neural Network model on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_classification_dialog(self.root, self.current_data, "neural_network")
            if result is not None:
                self.current_model, self.model_type, self.target_column, self.feature_columns = result
                self.update_status("Neural Network model trained successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train Neural Network model: {str(e)}")
    

    def perform_kmeans_clustering(self):
        """Perform K-means clustering on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_clustering_dialog(self.root, self.current_data, "kmeans")
            if result is not None:
                clustered_data, n_clusters, metrics = result
                self.current_data = clustered_data
                self.update_data_view()
                
                # Show metrics in a message box
                metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                messagebox.showinfo("Clustering Results", 
                                   f"K-means clustering completed with {n_clusters} clusters.\n\nMetrics:\n{metrics_text}")
                
                self.update_status(f"K-means clustering completed with {n_clusters} clusters.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform K-means clustering: {str(e)}")
    
    def perform_pam_clustering(self):
        """Perform PAM clustering on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            result = show_clustering_dialog(self.root, self.current_data, "pam")
            if result is not None:
                clustered_data, n_clusters, metrics = result
                self.current_data = clustered_data
                self.update_data_view()
                
                # Show metrics in a message box
                metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                messagebox.showinfo("Clustering Results", 
                                   f"PAM clustering completed with {n_clusters} clusters.\n\nMetrics:\n{metrics_text}")
                
                self.update_status(f"PAM clustering completed with {n_clusters} clusters.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform PAM clustering: {str(e)}")
    
    def perform_dbscan_clustering(self):
        """Perform DBSCAN clustering on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
        
        try:
            result = show_clustering_dialog(self.root, self.current_data, "dbscan")
            if result is not None:
                clustered_data, n_clusters, metrics = result
                self.current_data = clustered_data
                self.update_data_view()
                
                # Show metrics in a message box
                metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                messagebox.showinfo("Clustering Results", f"DBSCAN clustering completed with {n_clusters} clusters.\n\nMetrics:\n{metrics_text}")

                self.update_status(f"DBSCAN clustering completed with {n_clusters} clusters.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform DBSCAN clustering: {str(e)}")

    def perform_agnes_clustering(self):
        """Perform AGNES clustering on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
        
        try:
            result = show_clustering_dialog(self.root, self.current_data, "agnes")
            if result is not None:
                clustered_data, n_clusters, metrics = result
                self.current_data = clustered_data
                self.update_data_view()
                
                # Show metrics in a message box
                metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                messagebox.showinfo("Clustering Results", f"AGNES clustering completed with {n_clusters} clusters.\n\nMetrics:\n{metrics_text}")

                self.update_status(f"AGNES clustering completed with {n_clusters} clusters.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform AGNES clustering: {str(e)}")

    def perform_diana_clustering(self):
        """Perform DIANA clustering on the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
        
        try:
            result = show_clustering_dialog(self.root, self.current_data, "diana")
            if result is not None:
                clustered_data, n_clusters, metrics = result
                self.current_data = clustered_data
                self.update_data_view()
                
                # Show metrics in a message box
                metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                messagebox.showinfo("Clustering Results", f"DIANA clustering completed with {n_clusters} clusters.\n\nMetrics:\n{metrics_text}")

                self.update_status(f"DIANA clustering completed with {n_clusters} clusters.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform DIANA clustering: {str(e)}")

    def show_box_plot(self):
        """Show a box plot of the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            # Use the imported function
            create_box_plot(self.current_data)
            self.update_status("Box plot created.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create box plot: {str(e)}")
    
    def show_qq_plot(self):
        """Show a Q-Q plot of the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            # Use the imported function
            create_qq_plot(self.current_data)
            self.update_status("Q-Q plot created.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Q-Q plot: {str(e)}")
    
    def show_scatter_plot(self):
        """Show a scatter plot of the current dataset"""
        if self.current_data is None:
            messagebox.showinfo("No Data", "Please load a dataset first.")
            return
            
        try:
            # Use the imported function
            create_scatter_plot(self.current_data)
            self.update_status("Scatter plot created.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create scatter plot: {str(e)}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Use a themed Tkinter window
    app = DataVisualizationApp(root)
    root.mainloop()
