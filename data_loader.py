import pandas as pd
import numpy as np
import os

def load_dataset(file_path):
    """
    Load dataset from various formats (CSV, ARFF)
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return load_csv(file_path)
    elif file_extension == '.arff':
        return load_arff(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def load_csv(file_path):
    """
    Load dataset from CSV file
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def load_arff(file_path):
    """
    Load dataset from ARFF file
    """
    try:
        # Simple ARFF parser
        attributes = []
        data_section = False
        data_lines = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                if '@attribute' in line.lower():
                    # Extract attribute name
                    parts = line.split()
                    if len(parts) >= 2:
                        attr_name = parts[1].strip("'\"")
                        attributes.append(attr_name)
                
                if data_section:
                    if not line.startswith('@'):
                        data_lines.append(line)
                
                if '@data' in line.lower():
                    data_section = True
        
        # Process data lines
        rows = []
        for line in data_lines:
            if ',' in line:
                values = line.split(',')
                rows.append(values)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=attributes)
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # Keep as string if conversion fails
                
        return df
    except Exception as e:
        raise Exception(f"Error loading ARFF file: {str(e)}")
