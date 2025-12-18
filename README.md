# Data Visualization Tool

This project provides a comprehensive **data visualization**, **preprocessing**, and **machine learning** platform through an intuitive graphical user interface.

## Features

- **Data Loading**: Supports CSV, ARFF, and Excel file formats.
- **Data Preprocessing**: Includes cleaning, normalization, and handling of missing values.
- **Data Visualization**: Offers box plots, QQ plots, scatter plots, and more.
- **Classification Models**: Implements K-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, and Neural Networks.
- **Regression Models**: Supports Linear Regression and Neural Networks.
- **Clustering Algorithms**: Includes K-Means, PAM, DBSCAN, AGNES, and DIANA.
- **Model Evaluation**: Enables performance assessment and predictions for your datasets.

## Installation

Follow these steps to set up the environment and run the application:

1. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   ```
2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
3. **Install Dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

## Usage

To start the application, follow these steps:

1. Ensure the virtual environment is activated.
2. Run the following command to launch the application:
   ```bash
   python main.py
   ```

### Example Workflow

1. Launch the application.
2. Load a dataset in CSV, ARFF, or Excel format.
3. Preprocess the data (e.g., handle missing values or normalize).
4. Visualize the dataset using the built-in plotting tools.
5. Apply machine learning models for classification or regression.
6. Evaluate the model's performance and make predictions.

## Requirements

- **Python Version**: Ensure you have **Python 3.7+** installed.
- **Dependencies**: All required libraries are listed in the `requirements.txt` file.

To install the dependencies, run:
```bash
python -m pip install -r requirements.txt
```

## Notes

- For additional help or troubleshooting, refer to the documentation or contact the project maintainers.
- Contributions are welcome! Feel free to submit issues or pull requests.




//////////////
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python main.py