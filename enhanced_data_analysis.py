import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import scipy.stats as stats
import time
import shutil
from scipy import signal
import warnings
import base64
import matplotlib.gridspec as gridspec
import inspect
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from sklearn.model_selection import train_test_split  # Added to resolve the train_test_split import

# Type hint for calplot to help Pylance
# This tells the type checker that calplot exists even if not imported yet
try:
    import calplot  # type: ignore
except ImportError:
    calplot = None  # type: ignore

# Set matplotlib to use a faster non-interactive backend
plt.switch_backend('agg')

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set pandas options for performance
pd.options.mode.chained_assignment = None  # default='warn'

class EnhancedDataAnalyzer:
    """Advanced data analysis tool with comprehensive reporting and visualization"""
    
    def __init__(self, data_path=None, output_dir="OUTPUT", missing_threshold=0.3, n_jobs=-1):
        """Initialize the analyzer with data path, output directory, and parameters.
           n_jobs: Number of processes to use for parallel processing. -1 means use all available cores.
        """
        self.data_path = data_path
        self.missing_threshold = missing_threshold
        
        # Set number of cores for parallel processing
        self.n_jobs = n_jobs if n_jobs != -1 else max(1, multiprocessing.cpu_count() - 1)
        
        # Create timestamp for this analysis run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_time = time.time()  # Track analysis start time
        
        # Create a unique folder for this analysis run
        self.output_base_dir = Path(output_dir)
        self.output_dir = self.output_base_dir / f"Analysis_{self.timestamp}"
        
        # Create output directories
        self._create_output_dirs()
        
        # Initialize data attributes
        self.raw_data = None
        self.processed_data = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.modeling_results = {}
        self.time_series_results = {}
        
        # Cache for visualization data
        self._viz_cache = {}
        
        # Setup plotting style once at initialization
        self._setup_plotting_style()
    
    def _create_output_dirs(self):
        """Create output directories for storing analysis results."""
        # Create base output directory if it doesn't exist
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Create run-specific output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different analysis components - use fewer directories for efficiency
        subdirs = ["DataPreprocessing", "EDA", "Modeling"]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        logging.info(f"Output directories created at {self.output_dir}")
        return self.output_dir
    
    def _setup_plotting_style(self):
        """Configure matplotlib and seaborn for consistent, high-quality visualizations"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.format'] = 'png'
    
    def load_data(self):
        """Load data from the source file and detect column types"""
        logging.info(f"Loading data from {self.data_path}")
        
        try:
            # Detect file extension and load accordingly
            file_extension = os.path.splitext(self.data_path)[1].lower()
            
            # Optimize reading with appropriate engines and parameters
            if file_extension == '.csv':
                self.data = pd.read_csv(self.data_path, low_memory=False)
            elif file_extension in ['.xlsx', '.xls']:
                # For Excel, only read data and avoid reading formatting info
                self.data = pd.read_excel(self.data_path, engine='openpyxl')
            elif file_extension == '.json':
                self.data = pd.read_json(self.data_path, orient='records')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            # Make a copy of the original data - use efficient clone
            self.original_data = self.data.copy(deep=False)  # Faster shallow copy for read-only reference
            
            # Detect column types using vectorized operations
            self._detect_column_types()
            
            # Generate data loading report
            self._generate_data_loading_report()
            
            logging.info(f"Successfully loaded data with {len(self.data)} rows and {len(self.data.columns)} columns")
            return True
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return False
    
    def _detect_column_types(self):
        """Efficiently detect column types in the dataset using vectorized operations"""
        # Reset column type lists
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
        # Use pandas built-in type detection for faster classification
        dtypes = self.data.dtypes
        
        for col, dtype in dtypes.items():
            # Use pandas built-in type detection for efficiency
            if pd.api.types.is_numeric_dtype(dtype):
                self.numeric_cols.append(col)
            elif pd.api.types.is_datetime64_dtype(dtype):
                self.datetime_cols.append(col)
            else:
                # Check for convertible datetime columns (more efficient approach)
                try:
                    # Only try to convert first 1000 values for speed
                    sample = self.data[col].dropna().head(1000)
                    if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().all():
                        self.data[col] = pd.to_datetime(self.data[col])
                        self.datetime_cols.append(col)
                    else:
                        self.categorical_cols.append(col)
                except:
                    self.categorical_cols.append(col)
        
        logging.info(f"Detected {len(self.numeric_cols)} numeric columns, "
                    f"{len(self.categorical_cols)} categorical columns, and "
                    f"{len(self.datetime_cols)} datetime columns")
    
    def _generate_data_loading_report(self):
        """Generate a report on the loaded data"""
        report_path = self.output_dir / "DataPreprocessing" / f"data_loading_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA LOADING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Data Source: {self.data_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of Rows: {len(self.data)}\n")
            f.write(f"Number of Columns: {len(self.data.columns)}\n\n")
            
            f.write("COLUMN TYPES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Numeric Columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols)}\n")
            f.write(f"Categorical Columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols)}\n")
            f.write(f"Datetime Columns ({len(self.datetime_cols)}): {', '.join(self.datetime_cols)}\n\n")
            
            # Missing values summary
            missing_vals = self.data.isnull().sum()
            missing_percent = (missing_vals / len(self.data) * 100).round(2)
            
            f.write("MISSING VALUES SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            for col, count in missing_vals.items():
                if count > 0:
                    f.write(f"{col}: {count} values missing ({missing_percent[col]}%)\n")
            
            if missing_vals.sum() == 0:
                f.write("No missing values detected in the dataset.\n")
            else:
                f.write(f"\nTotal Missing Values: {missing_vals.sum()}\n")
                f.write(f"Average Missing Percentage: {(missing_vals.sum() / (len(self.data) * len(self.data.columns)) * 100):.2f}%\n")
            
            # Duplicate rows
            duplicate_count = self.data.duplicated().sum()
            f.write("\nDUPLICATE ROWS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of Duplicate Rows: {duplicate_count}\n")
            f.write(f"Percentage of Duplicate Rows: {(duplicate_count / len(self.data) * 100):.2f}%\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logging.info(f"Data loading report generated at {report_path}")
    
    def preprocess_data(self):
        """Preprocess the data by handling missing values, duplicates, and encoding"""
        if self.data is None:
            logging.error("No data loaded. Call load_data() first.")
            return False
        
        logging.info("Starting data preprocessing...")
        
        # Create a copy of the data for preprocessing
        self.processed_data = self.data.copy()
        
        # Handle duplicate rows - use faster inplace operations
        self._handle_duplicates()
        
        # Handle missing values - use faster vectorized operations
        self._handle_missing_values()
        
        # Encode categorical variables - use parallel processing for large datasets
        if len(self.categorical_cols) > 0:
            if len(self.categorical_cols) > 3 and len(self.processed_data) > 1000:
                # For large datasets, use parallel processing
                self._encode_categorical_parallel()
            else:
                # For smaller datasets, use regular processing
                self._encode_categorical_variables()
        
        # Generate preprocessing report
        self._generate_preprocessing_report()
        
        # Save preprocessed data
        preproc_data_path = self.output_dir / "DataPreprocessing" / f"preprocessed_data_{self.timestamp}.csv"
        self.processed_data.to_csv(preproc_data_path, index=False)
        logging.info(f"Preprocessed data saved to {preproc_data_path}")
        
        return True
    
    def _encode_categorical_parallel(self):
        """Encode categorical variables using parallel processing for large datasets"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Process each categorical column in parallel
            executor.map(self._encode_single_column, self.categorical_cols)
    
    def _encode_single_column(self, column):
        """Encode a single categorical column - used for parallel processing"""
        if column in self.processed_data.columns:
            # Create a more efficient label encoder
            unique_values = self.processed_data[column].dropna().unique()
            value_map = {val: i for i, val in enumerate(unique_values)}
            
            # Apply encoding
            encoded_col = f"{column}_encoded"
            self.processed_data[encoded_col] = self.processed_data[column].map(value_map)
            
            # Save encoding map for reference
            encoding_map_path = self.output_dir / "DataPreprocessing" / f"{column}_encoding_map_{self.timestamp}.txt"
            with open(encoding_map_path, 'w') as f:
                for val, code in value_map.items():
                    f.write(f"{val}: {code}\n")
            
            logging.info(f"Encoded {column} to {encoded_col}")
    
    def _handle_duplicates(self):
        """Handle duplicate rows in the dataset - optimized for large datasets"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return
        
        # Count duplicates before removal for reporting
        duplicate_count = self.processed_data.duplicated().sum()
        
        if duplicate_count > 0:
            # Store for reporting
            self.data_quality_metrics = getattr(self, 'data_quality_metrics', {})
            self.data_quality_metrics['duplicate_count'] = duplicate_count
            self.data_quality_metrics['duplicate_pct'] = (duplicate_count / len(self.processed_data)) * 100
            
            # Remove duplicates in-place for efficiency
            self.processed_data.drop_duplicates(inplace=True, keep='first')
            logging.info(f"Removing {duplicate_count} duplicate rows")
    
    def _handle_missing_values(self):
        """Handle missing values using appropriate methods for each column type"""
        # Check for columns with too many missing values
        missing_percent = self.processed_data.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > self.missing_threshold].index
        
        if len(cols_to_drop) > 0:
            logging.info(f"Dropping columns with >={self.missing_threshold*100}% missing values: {', '.join(cols_to_drop)}")
            self.processed_data = self.processed_data.drop(columns=cols_to_drop)
            
            # Update column type lists
            self.numeric_cols = [col for col in self.numeric_cols if col not in cols_to_drop]
            self.categorical_cols = [col for col in self.categorical_cols if col not in cols_to_drop]
            self.datetime_cols = [col for col in self.datetime_cols if col not in cols_to_drop]
        
        # Impute missing values
        for col in self.numeric_cols:
            if self.processed_data[col].isnull().sum() > 0:
                # Use median for numeric data (more robust to outliers than mean)
                median_val = self.processed_data[col].median()
                self.processed_data[col] = self.processed_data[col].fillna(median_val)
                logging.info(f"Filled missing values in {col} with median: {median_val}")
        
        for col in self.categorical_cols:
            if self.processed_data[col].isnull().sum() > 0:
                # Use mode for categorical data
                mode_val = self.processed_data[col].mode()[0]
                self.processed_data[col] = self.processed_data[col].fillna(mode_val)
                logging.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        for col in self.datetime_cols:
            if self.processed_data[col].isnull().sum() > 0:
                # Forward fill for time series data
                self.processed_data[col] = self.processed_data[col].fillna(method='ffill')
                # If still has NaN (e.g., at the start), use backward fill
                self.processed_data[col] = self.processed_data[col].fillna(method='bfill')
                logging.info(f"Filled missing values in {col} using forward/backward fill")
    
    def _encode_categorical_variables(self):
        """Encode categorical variables for analysis"""
        if not self.categorical_cols:
            return
            
        from sklearn.preprocessing import LabelEncoder
        
        # Save original categorical values for reporting
        original_categorical = {col: self.processed_data[col].unique() for col in self.categorical_cols}
        
        # Create label encoder for each categorical column
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Create a new column with the encoded values
            encoded_col = f"{col}_encoded"
            self.processed_data[encoded_col] = le.fit_transform(self.processed_data[col].astype(str))
            logging.info(f"Encoded {col} to {encoded_col}")
            
            # Add mapping to report
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            
            # Save encoding mapping
            mapping_path = self.output_dir / "DataPreprocessing" / f"{col}_encoding_map_{self.timestamp}.txt"
            with open(mapping_path, 'w') as f:
                f.write(f"Encoding mapping for {col}:\n\n")
                for original, encoded in mapping.items():
                    f.write(f"{original} -> {encoded}\n")
    
    def _generate_preprocessing_report(self):
        """Generate a detailed report on the preprocessing steps"""
        report_path = self.output_dir / "DataPreprocessing" / f"preprocessing_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA PREPROCESSING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Duplicates
            original_rows = len(self.data)
            processed_rows = len(self.processed_data)
            duplicates_removed = original_rows - processed_rows
            
            f.write("DUPLICATE REMOVAL\n")
            f.write("-" * 80 + "\n")
            f.write(f"Original row count: {original_rows}\n")
            f.write(f"Rows after duplicate removal: {processed_rows}\n")
            f.write(f"Duplicates removed: {duplicates_removed}\n")
            f.write(f"Percentage of duplicates: {(duplicates_removed / original_rows * 100):.2f}%\n\n")
            
            # Missing values
            f.write("MISSING VALUE HANDLING\n")
            f.write("-" * 80 + "\n")
            
            original_missing = self.data.isnull().sum().sum()
            processed_missing = self.processed_data.isnull().sum().sum()
            
            f.write(f"Original missing values: {original_missing}\n")
            f.write(f"Missing values after preprocessing: {processed_missing}\n")
            f.write(f"Missing values imputed: {original_missing - processed_missing}\n\n")
            
            # Columns dropped
            original_cols = set(self.data.columns)
            processed_cols = set(self.processed_data.columns)
            dropped_cols = original_cols - processed_cols
            
            f.write("COLUMNS DROPPED\n")
            f.write("-" * 80 + "\n")
            
            if dropped_cols:
                for col in dropped_cols:
                    missing_percent = (self.data[col].isnull().sum() / len(self.data) * 100).round(2)
                    f.write(f"{col}: {missing_percent}% missing values\n")
            else:
                f.write("No columns were dropped during preprocessing.\n\n")
            
            # Categorical encoding
            f.write("CATEGORICAL ENCODING\n")
            f.write("-" * 80 + "\n")
            
            if self.categorical_cols:
                for col in self.categorical_cols:
                    encoded_col = f"{col}_encoded"
                    if encoded_col in self.processed_data.columns:
                        f.write(f"Column '{col}' encoded as '{encoded_col}'\n")
                        f.write(f"Number of unique categories: {self.processed_data[col].nunique()}\n\n")
            else:
                f.write("No categorical variables were encoded.\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logging.info(f"Preprocessing report generated at {report_path}")

    def perform_eda(self):
        """Perform Exploratory Data Analysis with optimized processing"""
        if self.processed_data is None:
            logging.error("No processed data available. Call preprocess_data() first.")
            return False
        
        logging.info("Starting Exploratory Data Analysis...")
        logging.info("Calculating descriptive statistics...")
        
        # Create visualizations for categorical variables
        for col in self.categorical_cols:
            if col in self.processed_data.columns:
                self._plot_categorical_barplot(col)
        
        # Analyze distributions
        self._analyze_distributions()
        
        # Generate EDA report
        self._generate_eda_report()
        
        logging.info("EDA completed successfully")
        return True
    
    def _plot_categorical_barplot(self, column):
        """Create a bar plot for a categorical variable"""
        if column not in self.processed_data.columns:
            return
            
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Get value counts and sort for better visualization
        value_counts = self.processed_data[column].value_counts()
        
        # Plot the bar chart
        ax = sns.barplot(x=value_counts.index, y=value_counts.values)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add labels and title
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Distribution of {column}')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        self._save_figure(f"{column}_barplot", subdir="EDA")
    
    def _analyze_distributions(self):
        """Analyze the distributions of all variables"""
        # Analyze categorical variables distributions
        for col in self.categorical_cols:
            if col in self.processed_data.columns:
                self._analyze_categorical_distribution(col)
        
        # Analyze numeric variables distributions (if any)
        for col in self.numeric_cols:
            if col in self.processed_data.columns:
                self._analyze_numeric_distribution(col)
    
    def _analyze_categorical_distribution(self, column):
        """Analyze the distribution of a categorical variable"""
        if column not in self.processed_data.columns:
            return
            
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Get value counts
        value_counts = self.processed_data[column].value_counts()
        
        # Plot the distribution
        sns.countplot(y=self.processed_data[column], order=value_counts.index)
        
        # Add labels and title
        plt.xlabel('Count')
        plt.ylabel(column)
        plt.title(f'Distribution of {column}')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        self._save_figure(f"{column}_distribution")
        
        # Create a pie chart for better proportion visualization
        plt.figure(figsize=(8, 8))
        
        # Calculate percentages
        percentages = value_counts / value_counts.sum() * 100
        
        # Plot the pie chart
        plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f'Percentage Distribution of {column}')
        
        # Save the figure
        self._save_figure(f"{column}_pie_chart")
    
    def _analyze_numeric_distribution(self, column):
        """Analyze the distribution of a numeric variable"""
        if column not in self.processed_data.columns:
            return
            
        # Create the figure
        plt.figure(figsize=(12, 6))
        
        # Create a 1x2 subplot for histogram and boxplot
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        
        # Histogram with KDE
        ax0 = plt.subplot(gs[0])
        sns.histplot(self.processed_data[column].dropna(), kde=True, ax=ax0)
        ax0.set_title(f'Histogram of {column}')
        ax0.set_xlabel(column)
        ax0.set_ylabel('Frequency')
        
        # Boxplot
        ax1 = plt.subplot(gs[1])
        sns.boxplot(y=self.processed_data[column].dropna(), ax=ax1)
        ax1.set_title(f'Boxplot of {column}')
        ax1.set_ylabel(column)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        self._save_figure(f"{column}_distribution")
    
    def _generate_eda_report(self):
        """Generate a detailed EDA report"""
        report_path = self.output_dir / "EDA" / f"eda_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {os.path.basename(self.data_path)}\n")
            f.write(f"Number of Observations: {len(self.processed_data)}\n\n")
            
            # Descriptive Statistics
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-" * 80 + "\n")
            
            if 'numeric' in self.stats:
                f.write("\nNumeric Variables:\n\n")
                
                for col in self.numeric_cols:
                    stats = self.stats['numeric'][col]
                    f.write(f"Column: {col}\n")
                    f.write(f"  Count: {stats['count']:.0f}\n")
                    f.write(f"  Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Std Dev: {stats['std']:.4f}\n")
                    f.write(f"  Min: {stats['min']:.4f}\n")
                    f.write(f"  25%: {stats['25%']:.4f}\n")
                    f.write(f"  Median: {stats['50%']:.4f}\n")
                    f.write(f"  75%: {stats['75%']:.4f}\n")
                    f.write(f"  Max: {stats['max']:.4f}\n")
                    f.write(f"  Skewness: {stats['skewness']:.4f}\n")
                    f.write(f"  Kurtosis: {stats['kurtosis']:.4f}\n\n")
            
            if 'categorical' in self.stats:
                f.write("\nCategorical Variables:\n\n")
                
                for col in self.categorical_cols:
                    stats = self.stats['categorical'][col]
                    f.write(f"Column: {col}\n")
                    f.write(f"  Unique Values: {stats['unique_count']}\n")
                    f.write(f"  Most Common: {stats['top_value']} (appears {stats['top_count']} times, {stats['top_percentage']:.2f}%)\n")
                    
                    # Show top 5 categories
                    f.write("  Top Categories:\n")
                    for i, (value, count) in enumerate(stats['value_counts'].items()):
                        if i < 5:
                            percentage = (count / len(self.processed_data)) * 100
                            f.write(f"    {value}: {count} ({percentage:.2f}%)\n")
                        else:
                            break
                    f.write("\n")
            
            if 'datetime' in self.stats:
                f.write("\nDateTime Variables:\n\n")
                
                for col in self.datetime_cols:
                    stats = self.stats['datetime'][col]
                    f.write(f"Column: {col}\n")
                    f.write(f"  Min Date: {stats['min_date']}\n")
                    f.write(f"  Max Date: {stats['max_date']}\n")
                    f.write(f"  Range (days): {stats['range_days']}\n")
                    f.write(f"  Unique Values: {stats['unique_count']}\n\n")
            
            # Outlier Analysis
            if hasattr(self, 'outliers'):
                f.write("\nOUTLIER ANALYSIS\n")
                f.write("-" * 80 + "\n")
                
                for col, outlier_info in self.outliers.items():
                    f.write(f"\nColumn: {col}\n")
                    
                    # IQR method
                    iqr_info = outlier_info['iqr_method']
                    f.write("  IQR Method:\n")
                    f.write(f"    Lower Bound: {iqr_info['lower_bound']:.4f}\n")
                    f.write(f"    Upper Bound: {iqr_info['upper_bound']:.4f}\n")
                    f.write(f"    Outliers Found: {iqr_info['outlier_count']} ({iqr_info['outlier_percentage']:.2f}%)\n")
                    
                    # Z-score method
                    z_info = outlier_info['z_score_method']
                    f.write("  Z-Score Method (|z| > 3):\n")
                    f.write(f"    Outliers Found: {z_info['outlier_count']} ({z_info['outlier_percentage']:.2f}%)\n\n")
            
            # Correlation Analysis
            if hasattr(self, 'correlations'):
                f.write("\nCORRELATION ANALYSIS\n")
                f.write("-" * 80 + "\n")
                
                if hasattr(self, 'significant_correlations'):
                    f.write("\nSignificant Correlations:\n\n")
                    
                    if not self.significant_correlations:
                        f.write("  No significant correlations found (threshold: |r| > 0.5)\n\n")
                    else:
                        for corr in self.significant_correlations:
                            col1, col2 = corr['variables']
                            f.write(f"  {col1} vs {col2}:\n")
                            f.write(f"    Pearson: {corr['pearson']:.4f}\n")
                            f.write(f"    Spearman: {corr['spearman']:.4f}\n")
                            f.write(f"    Strength: {corr['strength']}\n")
                            f.write(f"    Direction: {corr['direction']}\n\n")
            
            # Distribution Analysis
            if hasattr(self, 'normality_tests'):
                f.write("\nDISTRIBUTION ANALYSIS\n")
                f.write("-" * 80 + "\n")
                
                f.write("\nNormality Tests:\n\n")
                
                for col, test_results in self.normality_tests.items():
                    f.write(f"  {col}:\n")
                    f.write(f"    Test: {test_results['test']}\n")
                    f.write(f"    Statistic: {test_results['statistic']:.4f}\n")
                    f.write(f"    p-value: {test_results['p_value']:.4f}\n")
                    f.write(f"    Normal Distribution: {'Yes' if test_results['is_normal'] else 'No'}\n\n")
            
            # Feature Importance
            if hasattr(self, 'feature_importance'):
                f.write("\nFEATURE IMPORTANCE ANALYSIS\n")
                f.write("-" * 80 + "\n")
                
                if 'mutual_info' in self.feature_importance:
                    mi_info = self.feature_importance['mutual_info']
                    f.write(f"\nMutual Information for predicting {mi_info['target']}:\n\n")
                    
                    # Sort features by importance
                    sorted_features = sorted(
                        mi_info['scores'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    for feature, score in sorted_features:
                        f.write(f"  {feature}: {score:.4f}\n")
            
            # Visualization References
            f.write("\n\nVISUALIZATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("The following visualizations have been generated in the EDA folder:\n\n")
            
            # List generated plots
            eda_dir = self.output_dir / "EDA"
            for file in eda_dir.glob(f"*{self.timestamp}*.png"):
                f.write(f"  - {file.name}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logging.info(f"EDA report generated at {report_path}")

    def _save_figure(self, filename, title=None, subdir=None):
        """Save the current matplotlib figure to a file with optimized settings."""
        # If filename is a path, use it directly; otherwise construct the path
        if not isinstance(filename, (str, Path)):
            raise ValueError("Filename must be a string or Path")
            
        # If filename is just a name without extension, add the timestamp and extension
        if isinstance(filename, str) and not filename.endswith('.png'):
            filename = f"{filename}_{self.timestamp}.png"
        
        # Determine the subdirectory to save to
        if subdir:
            output_dir = self.output_dir / subdir
        else:
            # Simplify directory determination - use fewer subdirectories
            if 'EDA' in inspect.stack()[1].function:
                output_dir = self.output_dir / "EDA"
            elif any(x in inspect.stack()[1].function for x in ['preprocessing', 'missing', 'duplicates']):
                output_dir = self.output_dir / "DataPreprocessing"
            elif any(x in inspect.stack()[1].function for x in ['modeling', 'cluster', 'regression', 'classification']):
                output_dir = self.output_dir / "Modeling"
            else:
                output_dir = self.output_dir
        
        # Make sure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct the full path
        if isinstance(filename, Path):
            filepath = filename
        else:
            filepath = output_dir / filename
        
        # Add title if provided
        if title:
            plt.suptitle(title)
            
        # Save the figure with optimized settings for better performance
        plt.savefig(filepath, dpi=150, bbox_inches='tight', format='png')
        plt.close('all')  # Close all figures to free memory
        
        logging.info(f"Figure saved to {filepath}")
        return filepath
    
    def perform_time_series_analysis(self):
        """Perform time series analysis on datetime columns."""
        if not self.datetime_cols:
            logging.info("No datetime columns found for time series analysis.")
            
            # Create empty directories and report for consistency
            ts_dir = self.output_dir / "TimeSeries"
            ts_dir.mkdir(exist_ok=True)
            
            # Generate a minimal report
            report_path = ts_dir / f"time_series_report_{self.timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("TIME SERIES ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write("No datetime columns found for time series analysis.\n")
                f.write("To perform time series analysis, include columns with datetime data type.\n\n")
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            # Create a placeholder visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No datetime columns available for time series analysis", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            self._save_figure("no_time_series_data", subdir="TimeSeries")
            
            return False
        
        # Create a directory for time series analysis
        ts_dir = self.output_dir / "TimeSeries"
        ts_dir.mkdir(exist_ok=True)
        
        # If no numeric columns, we can only do basic date analysis
        if not self.numeric_cols:
            logging.info("No numeric columns found for time series analysis.")
            
            # For each datetime column, at least visualize the distribution of dates
            for date_col in self.datetime_cols:
                # Create date distribution plot
                plt.figure(figsize=(12, 6))
                
                # Extract just the date component for cleaner visualization
                if pd.api.types.is_datetime64_any_dtype(self.processed_data[date_col]):
                    date_series = self.processed_data[date_col].dt.date
                    date_counts = self.processed_data.groupby(date_series).size()
                    
                    # Bar plot of date counts
                    plt.bar(date_counts.index, date_counts.values)
                    plt.title(f"Distribution of Records by {date_col}")
                    plt.xlabel("Date")
                    plt.ylabel("Number of Records")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save the plot
                    self._save_figure(f"{date_col}_distribution", subdir="TimeSeries")
                    
                    # Create a calendar heatmap if there are enough dates
                    if len(date_counts) >= 7:  # At least a week of data
                        self._create_calendar_heatmap(date_col)
            
            # Generate a report with basic date information
            self._generate_time_series_report()
            return True
            
        # Process each datetime column with all numeric columns
        for date_col in self.datetime_cols:
            if pd.api.types.is_datetime64_any_dtype(self.processed_data[date_col]):
                # For each numeric column, create time series visualizations
                for num_col in self.numeric_cols:
                    # Create a valid dataframe for time series analysis
                    ts_df = self.processed_data[[date_col, num_col]].dropna()
                    
                    # Only proceed if we have enough data points
                    if len(ts_df) >= 3:  # Minimum needed for basic trend
                        self._analyze_single_time_series(ts_df, num_col, date_col)
                    else:
                        logging.warning(f"Not enough data points for time series analysis of {num_col} by {date_col}")
                
                # Create calendar heatmap for this date column
                self._create_calendar_heatmap(date_col)
        
        # Generate time series report
        self._generate_time_series_report()
        return True
        
    def _create_calendar_heatmap(self, date_col):
        """Create a calendar heatmap visualization for a date column."""
        # Only create if we have enough dates to make it meaningful
        unique_dates = self.processed_data[date_col].dt.date.nunique()
        if unique_dates < 7:  # Not enough for a meaningful heatmap
            return
            
        # Create a date count series
        date_series = self.processed_data[date_col].dt.date
        date_counts = self.processed_data.groupby(date_series).size()
        
        # Convert to DataFrame with date index
        date_df = pd.DataFrame({'count': date_counts})
        date_df.index = pd.DatetimeIndex(date_df.index)
        
        # Get the min and max dates to determine the calendar range
        min_date = date_df.index.min()
        max_date = date_df.index.max()
        
        # Only proceed if we have at least a 7-day range
        if (max_date - min_date).days < 7:
            return
            
        # Create the calendar heatmap
        plt.figure(figsize=(12, 8))
        
        # If we have calplot, use it, otherwise use a custom approach
        try:
            # Use the global calplot variable defined at the top of the file
            if calplot is not None:
                calplot.calplot(date_df['count'], cmap='YlGnBu', 
                              yearascending=True, daylabels='MTWTFSS')
                plt.title(f"Calendar Heatmap of Records by {date_col}")
            else:
                # Fall back to a simple heatmap by day of week and week number
                date_df['day_of_week'] = date_df.index.dayofweek
                date_df['week_of_year'] = date_df.index.isocalendar().week
                
                # Pivot to create day-of-week vs week-of-year
                pivot_df = date_df.pivot_table(
                    index='day_of_week', 
                    columns='week_of_year', 
                    values='count', 
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Create heatmap
                sns.heatmap(pivot_df, cmap='YlGnBu', cbar_kws={'label': 'Record Count'})
                plt.title(f"Records by Week and Day of Week ({date_col})")
                plt.xlabel("Week of Year")
                plt.ylabel("Day of Week")
                plt.yticks(np.arange(7) + 0.5, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        except Exception as e:
            logging.warning(f"Error creating calendar heatmap: {str(e)}")
            # Fall back to a simple bar chart
            monthly_counts = date_df.resample('M')['count'].sum()
            plt.bar(monthly_counts.index, monthly_counts.values)
            plt.title(f"Monthly Record Counts for {date_col}")
            plt.xlabel("Month")
            plt.ylabel("Record Count")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_figure(f"{date_col}_calendar_heatmap", subdir="TimeSeries")
    
    def _analyze_single_time_series(self, ts_df, column, date_col):
        """Analyze a single time series (one numeric column by one date column)"""
        # Check the frequency (daily, monthly, etc.)
        ts_freq = self._detect_time_series_frequency(ts_df.index)
        
        # Calculate basic metrics
        basic_metrics = {
            'points': len(ts_df),
            'min': ts_df[column].min(),
            'max': ts_df[column].max(),
            'mean': ts_df[column].mean(),
            'median': ts_df[column].median(),
            'std': ts_df[column].std(),
            'frequency': ts_freq
        }
        
        # Plot the time series
        plt.figure(figsize=(12, 6))
        plt.plot(ts_df.index, ts_df[column])
        plt.title(f'Time Series: {column} by {date_col}')
        plt.xlabel(date_col)
        plt.ylabel(column)
        plt.grid(True)
        
        # Add trend line
        from scipy.signal import savgol_filter
        if len(ts_df) > 10:  # Enough data for smoothing
            try:
                # Use Savitzky-Golay filter for trend
                window_size = min(11, len(ts_df) - (len(ts_df) % 2 - 1))  # Must be odd and <= length
                if window_size > 2:  # Minimum 3 points required
                    trend = savgol_filter(ts_df[column], window_size, 1)
                    plt.plot(ts_df.index, trend, 'r--', label='Trend')
                    plt.legend()
            except Exception as e:
                logging.warning(f"Could not compute trend line: {str(e)}")
        
        ts_plot_path = self.output_dir / "TimeSeries" / f"ts_{column}_by_{date_col}_{self.timestamp}.png"
        self._save_figure(ts_plot_path)
        
        # Calculate moving averages if enough data
        moving_avgs = {}
        if len(ts_df) >= 10:
            window_sizes = [3, 7, 14, 30]
            for window in window_sizes:
                if len(ts_df) > window:
                    ma_col = f'MA_{window}'
                    ts_df[ma_col] = ts_df[column].rolling(window=window).mean()
                    moving_avgs[str(window)] = ts_df[ma_col].dropna().to_dict()
            
            # Plot moving averages
            if moving_avgs:
                plt.figure(figsize=(12, 6))
                plt.plot(ts_df.index, ts_df[column], label='Original')
                
                for window in window_sizes:
                    if len(ts_df) > window:
                        ma_col = f'MA_{window}'
                        plt.plot(ts_df.index, ts_df[ma_col], label=f'MA-{window}')
                
                plt.title(f'Moving Averages: {column} by {date_col}')
                plt.xlabel(date_col)
                plt.ylabel(column)
                plt.legend()
                plt.grid(True)
                
                ma_plot_path = self.output_dir / "TimeSeries" / f"ma_{column}_by_{date_col}_{self.timestamp}.png"
                self._save_figure(ma_plot_path)
        
        # Time series decomposition if enough data
        decomposition = {}
        if len(ts_df) >= 12:  # Need enough data points for decomposition
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Need to handle irregular time series by resampling
                if ts_freq:
                    # Convert to regular time series
                    regular_ts = ts_df[column].asfreq(ts_freq)
                    
                    # Fill gaps for decomposition
                    regular_ts = regular_ts.interpolate(method='linear')
                    
                    # Decompose the time series
                    result = seasonal_decompose(regular_ts, model='additive')
                    
                    # Plot decomposition
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
                    result.observed.plot(ax=ax1)
                    ax1.set_title('Observed')
                    ax1.set_xlabel('')
                    
                    result.trend.plot(ax=ax2)
                    ax2.set_title('Trend')
                    ax2.set_xlabel('')
                    
                    result.seasonal.plot(ax=ax3)
                    ax3.set_title('Seasonality')
                    ax3.set_xlabel('')
                    
                    result.resid.plot(ax=ax4)
                    ax4.set_title('Residuals')
                    
                    decomp_plot_path = self.output_dir / "TimeSeries" / f"decomp_{column}_by_{date_col}_{self.timestamp}.png"
                    self._save_figure(decomp_plot_path, title=f"Time Series Decomposition: {column}")
                    
                    # Store decomposition results
                    decomposition = {
                        'trend': result.trend.dropna().to_dict(),
                        'seasonal': result.seasonal.dropna().to_dict(),
                        'residual': result.resid.dropna().to_dict()
                    }
                    
                    # Check for seasonality
                    seasonal_strength = abs(result.seasonal).mean() / abs(result.observed).mean()
                    decomposition['seasonal_strength'] = float(seasonal_strength)
                    decomposition['has_seasonality'] = seasonal_strength > 0.1  # Arbitrary threshold
            except Exception as e:
                logging.warning(f"Could not perform time series decomposition: {str(e)}")
        
        # Autocorrelation analysis
        autocorrelation = {}
        if len(ts_df) > 10:
            try:
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                
                # Plot ACF and PACF
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                plot_acf(ts_df[column].dropna(), ax=ax1, lags=min(20, len(ts_df) // 2))
                ax1.set_title(f'Autocorrelation Function (ACF): {column}')
                
                plot_pacf(ts_df[column].dropna(), ax=ax2, lags=min(20, len(ts_df) // 2))
                ax2.set_title(f'Partial Autocorrelation Function (PACF): {column}')
                
                acf_plot_path = self.output_dir / "TimeSeries" / f"acf_pacf_{column}_by_{date_col}_{self.timestamp}.png"
                self._save_figure(acf_plot_path)
                
                # Calculate autocorrelation values
                from statsmodels.tsa.stattools import acf, pacf
                acf_values = acf(ts_df[column].dropna(), nlags=min(10, len(ts_df) // 2))
                pacf_values = pacf(ts_df[column].dropna(), nlags=min(10, len(ts_df) // 2))
                
                autocorrelation = {
                    'acf': acf_values.tolist(),
                    'pacf': pacf_values.tolist()
                }
                
                # Check for significant lags (arbitrary threshold)
                sig_lags = [i for i, val in enumerate(acf_values) if i > 0 and abs(val) > 0.3]
                autocorrelation['significant_lags'] = sig_lags
                autocorrelation['has_autocorrelation'] = len(sig_lags) > 0
            except Exception as e:
                logging.warning(f"Could not perform autocorrelation analysis: {str(e)}")
        
        # Forecasting with ARIMA or Prophet if enough data
        forecasting = {}
        if len(ts_df) >= 10:  # Need enough data points for forecasting
            try:
                # For this example, we'll use simple ARIMA
                # Prophet would require additional imports and setup
                from statsmodels.tsa.arima.model import ARIMA
                
                # Split data for validation (80% train, 20% test)
                train_size = int(len(ts_df) * 0.8)
                train = ts_df.iloc[:train_size]
                test = ts_df.iloc[train_size:]
                
                # Fit ARIMA model
                # Using simple parameters (1,1,1) for demonstration
                model = ARIMA(train[column], order=(1, 1, 1))
                model_fit = model.fit()
                
                # Forecast
                forecast_steps = len(test) if len(test) > 0 else 5
                forecast = model_fit.forecast(steps=forecast_steps)
                
                # Store forecasting results
                forecasting = {
                    'method': 'ARIMA(1,1,1)',
                    'forecast': forecast.to_dict(),
                    'confidence_intervals': {}
                }
                
                # Plot forecast
                plt.figure(figsize=(12, 6))
                plt.plot(ts_df.index[:train_size], ts_df[column][:train_size], label='Training Data')
                
                if len(test) > 0:
                    plt.plot(ts_df.index[train_size:], ts_df[column][train_size:], label='Test Data')
                
                # Plot forecast
                if len(test) > 0:
                    forecast_index = ts_df.index[train_size:train_size+forecast_steps]
                else:
                    # Create future dates based on the frequency
                    if ts_freq:
                        last_date = ts_df.index[-1]
                        forecast_index = pd.date_range(start=last_date, periods=forecast_steps+1, freq=ts_freq)[1:]
                    else:
                        forecast_index = range(train_size, train_size + forecast_steps)
                
                plt.plot(forecast_index, forecast, 'r--', label='Forecast')
                plt.title(f'ARIMA Forecast: {column} by {date_col}')
                plt.xlabel(date_col)
                plt.ylabel(column)
                plt.legend()
                plt.grid(True)
                
                forecast_plot_path = self.output_dir / "TimeSeries" / f"forecast_{column}_by_{date_col}_{self.timestamp}.png"
                self._save_figure(forecast_plot_path)
                
                # If validation data is available, calculate metrics
                if len(test) > 0:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    import math
                    
                    mae = mean_absolute_error(test[column], forecast[:len(test)])
                    rmse = math.sqrt(mean_squared_error(test[column], forecast[:len(test)]))
                    mape = np.mean(np.abs((test[column] - forecast[:len(test)]) / test[column])) * 100
                    
                    forecasting['metrics'] = {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape
                    }
            except Exception as e:
                logging.warning(f"Could not perform forecasting: {str(e)}")
        
        # Return all time series analysis results
        return {
            'basic_metrics': basic_metrics,
            'moving_averages': moving_avgs,
            'decomposition': decomposition,
            'autocorrelation': autocorrelation,
            'forecasting': forecasting
        }
    
    def _detect_time_series_frequency(self, date_index):
        """Detect the frequency of a time series based on the date index"""
        if len(date_index) < 2:
            return None
        
        # Calculate time differences
        time_diff = date_index.to_series().diff()[1:]
        
        # Get the most common difference
        if len(time_diff) == 0:
            return None
        
        median_diff = time_diff.median()
        days = median_diff.days
        
        # Determine frequency based on the median difference
        if days == 0:  # Less than a day
            hours = median_diff.seconds / 3600
            if hours <= 1:
                return 'H'  # Hourly
            elif hours <= 6:
                return '6H'  # 6-hourly
            else:
                return 'D'  # Daily
        elif 1 <= days < 2:
            return 'D'  # Daily
        elif 2 <= days < 8:
            return 'W'  # Weekly
        elif 8 <= days < 16:
            return '2W'  # Bi-weekly
        elif 16 <= days < 35:
            return 'M'  # Monthly
        elif 35 <= days < 100:
            return 'Q'  # Quarterly
        else:
            return 'A'  # Annual
    
    def _generate_time_series_report(self):
        """Generate a comprehensive time series analysis report."""
        # Create report file
        report_path = self.output_dir / "TimeSeries" / f"time_series_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TIME SERIES ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {os.path.basename(self.data_path)}\n\n")
            
            if not self.datetime_cols:
                f.write("No datetime columns found for time series analysis.\n")
                f.write("To perform time series analysis, include columns with datetime data type.\n\n")
            elif not self.numeric_cols:
                f.write("BASIC DATE ANALYSIS\n")
                f.write("-" * 80 + "\n\n")
                
                for date_col in self.datetime_cols:
                    f.write(f"Date Column: {date_col}\n")
                    
                    if pd.api.types.is_datetime64_any_dtype(self.processed_data[date_col]):
                        # Calculate date range and statistics
                        min_date = self.processed_data[date_col].min()
                        max_date = self.processed_data[date_col].max()
                        date_range = (max_date - min_date).days
                        unique_dates = self.processed_data[date_col].dt.date.nunique()
                        
                        f.write(f"  Start Date: {min_date.strftime('%Y-%m-%d')}\n")
                        f.write(f"  End Date: {max_date.strftime('%Y-%m-%d')}\n")
                        f.write(f"  Date Range: {date_range} days\n")
                        f.write(f"  Unique Dates: {unique_dates}\n")
                        
                        # Calculate distribution by weekday
                        weekday_counts = self.processed_data[date_col].dt.day_name().value_counts()
                        f.write("\n  Distribution by Day of Week:\n")
                        for day, count in weekday_counts.items():
                            f.write(f"    {day}: {count} records ({count/len(self.processed_data)*100:.1f}%)\n")
                        
                        f.write("\n  Note: No numeric columns available for full time series analysis.\n")
                        f.write("  To perform comprehensive time series analysis, include numeric columns.\n\n")
                    
                f.write("\n")
            else:
                # Check if we have time_series_results from previous analysis
                if hasattr(self, 'time_series_results') and self.time_series_results:
                    # Report for each date column
                    for date_col, date_results in self.time_series_results.items():
                        f.write(f"Analysis by Date Column: {date_col}\n")
                        f.write("-" * 80 + "\n\n")
                        
                        # For each numeric column
                        for num_col, col_results in date_results.items():
                            f.write(f"Numeric Variable: {num_col}\n")
                            f.write("  " + "-" * 50 + "\n\n")
                            
                            if 'trend' in col_results:
                                f.write(f"  Trend Analysis:\n")
                                f.write(f"    Trend Type: {col_results['trend']['type']}\n")
                                f.write(f"    Trend Strength: {col_results['trend']['strength']}\n")
                                f.write(f"    Direction: {col_results['trend']['direction']}\n\n")
                            
                            if 'seasonality' in col_results:
                                f.write(f"  Seasonality Analysis:\n")
                                f.write(f"    Detected: {col_results['seasonality']['detected']}\n")
                                if col_results['seasonality']['detected']:
                                    f.write(f"    Period: {col_results['seasonality']['period']}\n")
                                    f.write(f"    Strength: {col_results['seasonality']['strength']}\n")
                                f.write("\n")
                            
                            if 'statistics' in col_results:
                                f.write(f"  Key Statistics:\n")
                                stats = col_results['statistics']
                                f.write(f"    Mean: {stats['mean']:.4f}\n")
                                f.write(f"    Std Dev: {stats['std']:.4f}\n")
                                f.write(f"    Min: {stats['min']:.4f}\n")
                                f.write(f"    Max: {stats['max']:.4f}\n")
                                f.write(f"    Range: {stats['range']:.4f}\n\n")
                            
                            if 'forecast_potential' in col_results:
                                f.write(f"  Forecasting Potential:\n")
                                forecast = col_results['forecast_potential']
                                f.write(f"    Rating: {forecast['rating']}\n")
                                f.write(f"    Recommended Models: {', '.join(forecast['models'])}\n")
                                f.write(f"    Data Points: {forecast['data_points']}\n\n")
                            
                            f.write("\n")
                else:
                    # Basic information about date columns
                    f.write("BASIC DATE ANALYSIS\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for date_col in self.datetime_cols:
                        f.write(f"Date Column: {date_col}\n")
                        
                        if pd.api.types.is_datetime64_any_dtype(self.processed_data[date_col]):
                            # Calculate date range and statistics
                            min_date = self.processed_data[date_col].min()
                            max_date = self.processed_data[date_col].max()
                            date_range = (max_date - min_date).days
                            unique_dates = self.processed_data[date_col].dt.date.nunique()
                            
                            f.write(f"  Start Date: {min_date.strftime('%Y-%m-%d')}\n")
                            f.write(f"  End Date: {max_date.strftime('%Y-%m-%d')}\n")
                            f.write(f"  Date Range: {date_range} days\n")
                            f.write(f"  Unique Dates: {unique_dates}\n")
                            
                            # Calculate distribution by weekday
                            weekday_counts = self.processed_data[date_col].dt.day_name().value_counts()
                            f.write("\n  Distribution by Day of Week:\n")
                            for day, count in weekday_counts.items():
                                f.write(f"    {day}: {count} records ({count/len(self.processed_data)*100:.1f}%)\n")
            
            # Visualizations reference
            f.write("\nVISUALIZATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("The following visualizations have been generated:\n\n")
            
            # List plots in the TimeSeries directory
            ts_dir = self.output_dir / "TimeSeries"
            if ts_dir.exists():
                for file in ts_dir.glob(f"*{self.timestamp}*.png"):
                    f.write(f"  - {file.name}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
            
        logging.info(f"Time series report generated at {report_path}")

    def perform_predictive_modeling(self, target_col=None, model_type=None, test_size=0.2):
        """Perform predictive modeling on the dataset."""
        # Create directory for modeling outputs
        modeling_dir = self.output_dir / "Modeling"
        modeling_dir.mkdir(exist_ok=True)
        
        # Validate that we have enough data for modeling
        if self.processed_data is None or len(self.processed_data) < 20:
            logging.warning("Not enough data for reliable predictive modeling. Need at least 20 samples.")
            
            # Create a placeholder visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Insufficient data for modeling (have {len(self.processed_data) if self.processed_data is not None else 0} samples, need 20+)", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            self._save_figure("insufficient_data_for_modeling", subdir="Modeling")
            
            # Create a simple report
            report_path = modeling_dir / f"modeling_report_{self.timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("PREDICTIVE MODELING REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {os.path.basename(self.data_path)}\n\n")
                f.write("INSUFFICIENT DATA FOR MODELING\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"The dataset contains {len(self.processed_data) if self.processed_data is not None else 0} samples.\n")
                f.write("For reliable predictive modeling, at least 20 samples are recommended.\n\n")
                f.write("Recommendations:\n")
                f.write("1. Collect more data before attempting predictive modeling\n")
                f.write("2. Consider simpler statistical analysis methods for small datasets\n")
                f.write("3. Use bootstrapping or other resampling techniques if modeling is required\n\n")
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            # Initialize empty modeling results
            self.modeling_results = {
                'status': 'insufficient_data',
                'message': 'Not enough data for reliable predictive modeling. Need at least 20 samples.',
                'sample_count': len(self.processed_data) if self.processed_data is not None else 0
            }
            
            return False
        
        # Also verify we have numeric columns for modeling
        if not self.numeric_cols:
            logging.warning("No numeric columns available for predictive modeling.")
            
            # Create a placeholder visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No numeric columns available for predictive modeling", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            self._save_figure("no_numeric_data_for_modeling", subdir="Modeling")
            
            # Create a simple report
            report_path = modeling_dir / f"modeling_report_{self.timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("PREDICTIVE MODELING REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {os.path.basename(self.data_path)}\n\n")
                f.write("NO NUMERIC COLUMNS FOR MODELING\n")
                f.write("-" * 80 + "\n\n")
                f.write("Predictive modeling requires numeric columns for features and/or targets.\n")
                f.write("The current dataset does not contain any numeric columns.\n\n")
                f.write("Recommendations:\n")
                f.write("1. Add numeric measurements or metrics to your dataset\n")
                f.write("2. Consider encoding categorical variables as numeric features\n")
                f.write("3. Explore text-based or categorical-only analysis methods\n\n")
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            # Initialize empty modeling results
            self.modeling_results = {
                'status': 'no_numeric_data',
                'message': 'No numeric columns available for predictive modeling.',
                'categorical_cols': self.categorical_cols
            }
            
            # Create visualization of the categorical data instead
            if self.categorical_cols:
                self._visualize_categorical_relationships()
            
            return False
        
        # If target column not provided, auto-select based on context
        if target_col is None:
            # Try to select a good target column
            if len(self.numeric_cols) == 1:
                target_col = self.numeric_cols[0]
            elif 'target' in self.processed_data.columns:
                target_col = 'target'
            elif any(col.lower().endswith(('price', 'cost', 'value', 'amount')) for col in self.numeric_cols):
                # Find columns related to price/cost/value
                price_cols = [col for col in self.numeric_cols 
                           if col.lower().endswith(('price', 'cost', 'value', 'amount'))]
                target_col = price_cols[0]
            elif self.numeric_cols:
                # Just take the last numeric column as a default
                target_col = self.numeric_cols[-1]
        
        # Make sure target_col is valid
        if target_col not in self.processed_data.columns:
            logging.warning(f"Target column '{target_col}' not found in dataset. Skipping modeling.")
            
            # Initialize empty modeling results
            self.modeling_results = {
                'status': 'invalid_target',
                'message': f"Target column '{target_col}' not found in dataset.",
                'available_columns': list(self.processed_data.columns)
            }
            
            return False
            
        # Auto-detect problem type if not specified
        if model_type is None:
            if pd.api.types.is_numeric_dtype(self.processed_data[target_col]):
                model_type = 'regression'
            elif self.processed_data[target_col].nunique() <= 10:
                model_type = 'classification'
            else:
                model_type = 'clustering'
                
        # Initialize modeling results
        self.modeling_results = {
            'problem_type': model_type,
            'target_column': target_col,
            'models': {},
            'best_model': None,
            'feature_importance': {}
        }
        
        # Prepare features and target
        if model_type != 'clustering':
            # For supervised learning, separate features and target
            feature_cols = [col for col in self.numeric_cols if col != target_col]
            
            # If no feature columns after removing target, create dummy features
            if not feature_cols:
                # Create dummy features from categorical columns
                logging.info("No numeric feature columns found. Creating features from categorical variables.")
                for cat_col in self.categorical_cols:
                    # Skip if too many categories (would create too many features)
                    if self.processed_data[cat_col].nunique() <= 20:
                        dummy_df = pd.get_dummies(self.processed_data[cat_col], prefix=cat_col)
                        # Add to processed_data
                        self.processed_data = pd.concat([self.processed_data, dummy_df], axis=1)
                        # Update feature_cols
                        feature_cols.extend(dummy_df.columns.tolist())
            
            # If still no features, create synthetic ones
            if not feature_cols:
                logging.warning("No features available for modeling. Creating synthetic features.")
                # Create random features for demonstration
                np.random.seed(42)
                self.processed_data['synthetic_feature_1'] = np.random.normal(0, 1, len(self.processed_data))
                self.processed_data['synthetic_feature_2'] = np.random.normal(0, 1, len(self.processed_data))
                feature_cols = ['synthetic_feature_1', 'synthetic_feature_2']
            
            # Store features in modeling results
            self.modeling_results['features'] = feature_cols
            
            # Get data
            X = self.processed_data[feature_cols].copy()
            y = self.processed_data[target_col].copy()
            
            # If there are NaN values, handle them
            if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                logging.info("Handling missing values for modeling...")
                X = X.fillna(X.mean())
                if pd.api.types.is_numeric_dtype(y):
                    y = y.fillna(y.mean())
                else:
                    y = y.fillna(y.mode()[0])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Conduct appropriate modeling based on problem type
            if model_type == 'regression':
                self._perform_regression_modeling(X_train, X_test, y_train, y_test, feature_cols)
            elif model_type == 'classification':
                self._perform_classification_modeling(X_train, X_test, y_train, y_test, feature_cols)
        else:
            # For unsupervised learning (clustering), just use all numeric columns
            feature_cols = self.numeric_cols
            self.modeling_results['features'] = feature_cols
            
            # Get data
            X = self.processed_data[feature_cols].copy()
            
            # Handle missing values
            if X.isnull().sum().sum() > 0:
                X = X.fillna(X.mean())
            
            # Perform clustering
            self._perform_clustering_analysis(X, feature_cols)
        
        # Generate modeling report
        self._generate_modeling_report()
        
        return True
        
    def _visualize_categorical_relationships(self):
        """Create visualizations to explore relationships between categorical variables."""
        # Create directory for modeling outputs
        modeling_dir = self.output_dir / "Modeling"
        
        # Only proceed if we have at least 2 categorical columns
        if len(self.categorical_cols) < 2:
            return
            
        # Create heatmap of categorical co-occurrence (contingency tables)
        for i, col1 in enumerate(self.categorical_cols[:-1]):
            for col2 in self.categorical_cols[i+1:]:
                # Create contingency table
                cont_table = pd.crosstab(
                    self.processed_data[col1], 
                    self.processed_data[col2],
                    normalize='index'  # Normalize by rows
                )
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(cont_table, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Proportion'})
                plt.title(f"Relationship between {col1} and {col2}")
                plt.tight_layout()
                
                # Save figure
                self._save_figure(f"{col1}_vs_{col2}_heatmap", subdir="Modeling")
        
        # Create a mosaic plot for the main categorical variables (limit to 3 to avoid overcrowding)
        if len(self.categorical_cols) >= 2:
            plot_cols = self.categorical_cols[:3] if len(self.categorical_cols) > 3 else self.categorical_cols
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Use mosaic plot if available or fallback to pairplot
            try:
                from statsmodels.graphics.mosaicplot import mosaic
                
                # Prepare data
                plot_data = self.processed_data[plot_cols].copy()
                
                # Convert to string to avoid potential issues
                for col in plot_cols:
                    plot_data[col] = plot_data[col].astype(str)
                
                # Create dictionary of categories
                mosaic_data = {}
                for _, row in plot_data.iterrows():
                    key = tuple(row[plot_cols].values)
                    mosaic_data[key] = mosaic_data.get(key, 0) + 1
                
                # Create mosaic plot
                mosaic(mosaic_data, title="Categorical Relationships Mosaic Plot")
            except:
                # Fallback to simpler visualization
                g = sns.PairGrid(self.processed_data[plot_cols])
                g.map_diag(sns.countplot)
                g.map_offdiag(sns.countplot)
                plt.suptitle("Categorical Relationships", y=1.02)
            
            plt.tight_layout()
            self._save_figure("categorical_relationships_mosaic", subdir="Modeling")
        
        # Create stacked bar charts for each categorical variable with datetime
        if self.datetime_cols:
            date_col = self.datetime_cols[0]  # Use the first datetime column
            
            # Convert to period for easier grouping (e.g., by month)
            if pd.api.types.is_datetime64_any_dtype(self.processed_data[date_col]):
                # Try to determine a good period frequency based on date range
                date_range = (self.processed_data[date_col].max() - self.processed_data[date_col].min()).days
                
                if date_range <= 31:  # Less than a month
                    period_freq = 'D'  # Daily
                    period_name = "Day"
                elif date_range <= 365:  # Less than a year
                    period_freq = 'M'  # Monthly
                    period_name = "Month"
                else:
                    period_freq = 'Q'  # Quarterly
                    period_name = "Quarter"
                
                # Create period column
                period_col = f"{date_col}_{period_name}"
                self.processed_data[period_col] = self.processed_data[date_col].dt.to_period(period_freq)
                
                # For each categorical column, create a stacked bar chart by period
                for cat_col in self.categorical_cols:
                    # Create crosstab
                    ct = pd.crosstab(self.processed_data[period_col], self.processed_data[cat_col])
                    
                    # Plot stacked bar chart
                    plt.figure(figsize=(12, 8))
                    ct.plot(kind='bar', stacked=True, ax=plt.gca())
                    plt.title(f"{cat_col} by {period_name}")
                    plt.xlabel(period_name)
                    plt.ylabel("Count")
                    plt.legend(title=cat_col)
                    plt.tight_layout()
                    
                    # Save figure
                    self._save_figure(f"{cat_col}_by_{period_name.lower()}", subdir="Modeling")
    
    def _perform_regression_modeling(self, X_train, X_test, y_train, y_test, feature_cols):
        """Perform regression modeling with multiple algorithms"""
        logging.info("Performing regression modeling...")
        
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # Define the regression models to try
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Fit each model and evaluate
        best_r2 = -float('inf')
        best_model_name = None
        
        for name, model in regression_models.items():
            logging.info(f"Training {name}...")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.modeling_results['models'][name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': model  # Store the model object
            }
            
            # Check if this is the best model so far
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.modeling_results['feature_importance'][name] = dict(
                    zip(feature_cols, model.feature_importances_)
                )
            elif hasattr(model, 'coef_'):
                # For linear models
                self.modeling_results['feature_importance'][name] = dict(
                    zip(feature_cols, abs(model.coef_))
                )
            
            logging.info(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Visualize predictions vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{name} - Actual vs Predicted')
            plt.grid(True)
            
            pred_plot_path = self.output_dir / "Modeling" / f"regression_{name.replace(' ', '_')}_{self.timestamp}.png"
            self._save_figure(pred_plot_path)
            
            # Visualize residuals
            plt.figure(figsize=(10, 6))
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title(f'{name} - Residual Plot')
            plt.grid(True)
            
            resid_plot_path = self.output_dir / "Modeling" / f"residuals_{name.replace(' ', '_')}_{self.timestamp}.png"
            self._save_figure(resid_plot_path)
        
        # Record the best model
        self.modeling_results['best_model'] = best_model_name
        logging.info(f"Best regression model: {best_model_name} (R² = {best_r2:.4f})")
        
        # Feature importance visualization (for the best model)
        if best_model_name in self.modeling_results['feature_importance']:
            importances = self.modeling_results['feature_importance'][best_model_name]
            
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            features = [x[0] for x in sorted_features]
            scores = [x[1] for x in sorted_features]
            
            plt.figure(figsize=(10, 8))
            plt.barh(features, scores)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.tight_layout()
            
            fi_plot_path = self.output_dir / "Modeling" / f"feature_importance_{best_model_name.replace(' ', '_')}_{self.timestamp}.png"
            self._save_figure(fi_plot_path)

    def _perform_classification_modeling(self, X_train, X_test, y_train, y_test, feature_cols):
        """Perform classification modeling with multiple algorithms"""
        logging.info("Performing classification modeling...")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.preprocessing import label_binarize
        
        # Check if we need to binarize targets for metrics
        classes = np.unique(y_train)
        num_classes = len(classes)
        is_binary = num_classes == 2
        
        logging.info(f"Classification problem with {num_classes} classes")
        
        # Define the classification models to try
        if num_classes == 2:  # Binary classification
            classification_models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }
        else:  # Multi-class classification
            classification_models = {
                'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
        
        # Fit each model and evaluate
        best_f1 = -float('inf')
        best_model_name = None
        
        for name, model in classification_models.items():
            logging.info(f"Training {name}...")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            if is_binary:
                # Binary classification metrics
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            else:
                # Multi-class metrics (weighted average)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            self.modeling_results['models'][name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
                'model': model  # Store the model object
            }
            
            # Check if this is the best model so far
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.modeling_results['feature_importance'][name] = dict(
                    zip(feature_cols, model.feature_importances_)
                )
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    # Binary classification
                    self.modeling_results['feature_importance'][name] = dict(
                        zip(feature_cols, np.abs(model.coef_))
                    )
                else:
                    # Multi-class classification (average across classes)
                    self.modeling_results['feature_importance'][name] = dict(
                        zip(feature_cols, np.mean(np.abs(model.coef_), axis=0))
                    )
            
            logging.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, " +
                         f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Visualize confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'{name} - Confusion Matrix')
            
            cm_plot_path = self.output_dir / "Modeling" / f"confusion_matrix_{name.replace(' ', '_')}_{self.timestamp}.png"
            self._save_figure(cm_plot_path)
            
            # ROC curve for binary classification
            if is_binary and hasattr(model, 'predict_proba'):
                from sklearn.metrics import roc_curve, auc
                
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name} - ROC Curve')
                plt.legend(loc="lower right")
                
                roc_plot_path = self.output_dir / "Modeling" / f"roc_curve_{name.replace(' ', '_')}_{self.timestamp}.png"
                self._save_figure(roc_plot_path)
                
                # Add AUC to results
                self.modeling_results['models'][name]['roc_auc'] = roc_auc
        
        # Record the best model
        self.modeling_results['best_model'] = best_model_name
        logging.info(f"Best classification model: {best_model_name} (F1 = {best_f1:.4f})")
        
        # Feature importance visualization (for the best model)
        if best_model_name in self.modeling_results['feature_importance']:
            importances = self.modeling_results['feature_importance'][best_model_name]
            
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            features = [x[0] for x in sorted_features]
            scores = [x[1] for x in sorted_features]
            
            plt.figure(figsize=(10, 8))
            plt.barh(features, scores)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.tight_layout()
            
            fi_plot_path = self.output_dir / "Modeling" / f"feature_importance_{best_model_name.replace(' ', '_')}_{self.timestamp}.png"
            self._save_figure(fi_plot_path)
    
    def _perform_clustering_analysis(self, X, feature_cols):
        """Perform clustering analysis on the dataset"""
        logging.info("Performing clustering analysis...")
        
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        
        # Normalize the data for better clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize clustering results
        self.modeling_results['models'] = {}
        
        # K-Means Clustering
        # Determine optimal k using elbow method
        inertia_values = []
        silhouette_scores = []
        max_clusters = min(10, len(X) // 5)  # Don't try too many clusters
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia_values.append(kmeans.inertia_)
            
            # Calculate silhouette score
            labels = kmeans.labels_
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertia_values, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal k')
        plt.grid(True)
        
        elbow_plot_path = self.output_dir / "Modeling" / f"kmeans_elbow_method_{self.timestamp}.png"
        self._save_figure(elbow_plot_path)
        
        # Determine optimal k using both methods
        # Use k with highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logging.info(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        # Apply K-Means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(X_scaled)
        cluster_labels = kmeans.labels_
        
        # Store results
        self.modeling_results['models']['KMeans'] = {
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_scores[optimal_k - 2],  # -2 because k starts at 2
            'inertia': kmeans.inertia_,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_sizes': np.bincount(cluster_labels).tolist(),
            'labels': cluster_labels.tolist()
        }
        
        # Add cluster labels to processed data for later use
        self.processed_data['kmeans_cluster'] = cluster_labels
        
        # Visualize clusters
        # Use PCA for dimensionality reduction if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pca_explained_var = pca.explained_variance_ratio_.sum()
            
            # Store PCA results
            self.modeling_results['models']['PCA'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': pca_explained_var,
                'components': pca.components_.tolist()
            }
            
            # Plot clusters in 2D PCA space
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
            plt.scatter(
                pca.transform(kmeans.cluster_centers_)[:, 0],
                pca.transform(kmeans.cluster_centers_)[:, 1],
                marker='x', s=200, linewidths=3, color='r'
            )
            plt.colorbar(scatter)
            plt.title(f'K-Means Clustering (k={optimal_k}) - PCA Visualization')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True)
            
            kmeans_plot_path = self.output_dir / "Modeling" / f"kmeans_clusters_pca_{self.timestamp}.png"
            self._save_figure(kmeans_plot_path)
        else:
            # If already 2D, plot directly
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
            plt.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='r'
            )
            plt.colorbar(scatter)
            plt.title(f'K-Means Clustering (k={optimal_k})')
            plt.xlabel(feature_cols[0])
            plt.ylabel(feature_cols[1])
            plt.grid(True)
            
            kmeans_plot_path = self.output_dir / "Modeling" / f"kmeans_clusters_{self.timestamp}.png"
            self._save_figure(kmeans_plot_path)
        
        # Try DBSCAN clustering as well
        try:
            # DBSCAN doesn't require specifying number of clusters
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            # Count clusters (excluding noise points labeled as -1)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            noise_points = np.sum(dbscan_labels == -1)
            
            logging.info(f"DBSCAN found {n_clusters} clusters and {noise_points} noise points")
            
            if n_clusters > 1:  # DBSCAN found some clusters
                # Calculate silhouette score (ignoring noise points)
                if noise_points < len(X):  # Make sure we have non-noise points
                    non_noise_idx = dbscan_labels != -1
                    if np.sum(non_noise_idx) > 1:  # Need at least 2 non-noise points
                        sil_score = silhouette_score(
                            X_scaled[non_noise_idx], 
                            dbscan_labels[non_noise_idx]
                        )
                    else:
                        sil_score = 0
                else:
                    sil_score = 0
                
                # Store DBSCAN results
                self.modeling_results['models']['DBSCAN'] = {
                    'n_clusters': n_clusters,
                    'noise_points': int(noise_points),
                    'noise_percentage': float(noise_points / len(X) * 100),
                    'silhouette_score': sil_score,
                    'cluster_sizes': np.bincount(dbscan_labels[dbscan_labels != -1]).tolist(),
                    'labels': dbscan_labels.tolist()
                }
                
                # Add DBSCAN labels to processed data
                self.processed_data['dbscan_cluster'] = dbscan_labels
                
                # Visualize DBSCAN clusters
                if X.shape[1] > 2:
                    # Use the same PCA transformation as before
                    plt.figure(figsize=(10, 8))
                    colors = np.array(['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff',
                                       '#990000', '#009900', '#000099', '#999900', '#009999', '#990099'])
                    # Use red for noise points and other colors for clusters
                    color_idx = np.zeros(len(dbscan_labels), dtype=int)
                    color_idx[dbscan_labels == -1] = 0  # Noise points are red
                    for i, label in enumerate(np.unique(dbscan_labels[dbscan_labels != -1])):
                        color_idx[dbscan_labels == label] = i + 1
                    
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors[color_idx % len(colors)], alpha=0.6)
                    plt.title('DBSCAN Clustering - PCA Visualization')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    plt.grid(True)
                    
                    # Add a legend
                    from matplotlib.lines import Line2D
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Noise ({noise_points} points)',
                                             markerfacecolor='#ff0000', markersize=10)]
                    
                    for i, label in enumerate(np.unique(dbscan_labels[dbscan_labels != -1])):
                        cluster_size = np.sum(dbscan_labels == label)
                        legend_elements.append(
                            Line2D([0], [0], marker='o', color='w',
                                   label=f'Cluster {i+1} ({cluster_size} points)',
                                   markerfacecolor=colors[(i+1) % len(colors)], markersize=10)
                        )
                    
                    plt.legend(handles=legend_elements, title="Clusters")
                    
                    dbscan_plot_path = self.output_dir / "Modeling" / f"dbscan_clusters_pca_{self.timestamp}.png"
                    self._save_figure(dbscan_plot_path)
        except Exception as e:
            logging.warning(f"DBSCAN clustering failed: {str(e)}")
    
    def _generate_modeling_report(self):
        """Generate a comprehensive modeling report"""
        if not hasattr(self, 'modeling_results'):
            logging.warning("No modeling results to report")
            return
        
        report_path = self.output_dir / "Modeling" / f"modeling_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PREDICTIVE MODELING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {os.path.basename(self.data_path)}\n\n")
            
            # Basic modeling information
            problem_type = self.modeling_results['problem_type']
            target_col = self.modeling_results['target_column']
            features = self.modeling_results['features']
            
            f.write(f"Problem Type: {problem_type.upper()}\n")
            f.write(f"Target Variable: {target_col}\n")
            f.write(f"Number of Features: {len(features)}\n")
            f.write(f"Features: {', '.join(features)}\n\n")
            
            # Model evaluation
            f.write("MODEL EVALUATION\n")
            f.write("-" * 80 + "\n\n")
            
            # Different metrics based on problem type
            if problem_type == 'regression':
                f.write("Regression Metrics:\n\n")
                f.write(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12}\n")
                f.write("-" * 60 + "\n")
                
                for name, results in self.modeling_results['models'].items():
                    if 'rmse' in results:
                        f.write(f"{name:<25} {results['rmse']:<12.4f} {results['mae']:<12.4f} {results['r2']:<12.4f}\n")
                
                f.write("\n")
                best_model = self.modeling_results['best_model']
                if best_model:
                    f.write(f"Best Model: {best_model} (R² = {self.modeling_results['models'][best_model]['r2']:.4f})\n\n")
            
            elif problem_type == 'classification':
                f.write("Classification Metrics:\n\n")
                f.write(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
                if any('roc_auc' in results for results in self.modeling_results['models'].values()):
                    f.write(f" {'ROC AUC':<12}")
                f.write("\n")
                f.write("-" * 80 + "\n")
                
                for name, results in self.modeling_results['models'].items():
                    if 'accuracy' in results:
                        line = f"{name:<25} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
                        line += f"{results['recall']:<12.4f} {results['f1']:<12.4f}"
                        if 'roc_auc' in results:
                            line += f" {results['roc_auc']:<12.4f}"
                        f.write(line + "\n")
                
                f.write("\n")
                best_model = self.modeling_results['best_model']
                if best_model:
                    f.write(f"Best Model: {best_model} (F1 = {self.modeling_results['models'][best_model]['f1']:.4f})\n\n")
                    
                    # Confusion Matrix for the best model
                    f.write(f"Confusion Matrix for {best_model}:\n\n")
                    cm = self.modeling_results['models'][best_model]['confusion_matrix']
                    for row in cm:
                        f.write("  ".join(f"{cell:4d}" for cell in row) + "\n")
                    f.write("\n")
            
            elif problem_type == 'clustering':
                f.write("Clustering Results:\n\n")
                
                if 'KMeans' in self.modeling_results['models']:
                    kmeans = self.modeling_results['models']['KMeans']
                    f.write("K-Means Clustering:\n")
                    f.write(f"  Number of Clusters: {kmeans['n_clusters']}\n")
                    f.write(f"  Silhouette Score: {kmeans['silhouette_score']:.4f}\n")
                    f.write(f"  Inertia: {kmeans['inertia']:.4f}\n")
                    f.write("  Cluster Sizes: ")
                    for i, size in enumerate(kmeans['cluster_sizes']):
                        f.write(f"Cluster {i}: {size} samples, ")
                    f.write("\n\n")
                
                if 'DBSCAN' in self.modeling_results['models']:
                    dbscan = self.modeling_results['models']['DBSCAN']
                    f.write("DBSCAN Clustering:\n")
                    f.write(f"  Number of Clusters: {dbscan['n_clusters']}\n")
                    f.write(f"  Noise Points: {dbscan['noise_points']} ({dbscan['noise_percentage']:.2f}%)\n")
                    if 'silhouette_score' in dbscan:
                        f.write(f"  Silhouette Score: {dbscan['silhouette_score']:.4f}\n")
                    f.write("  Cluster Sizes (excluding noise): ")
                    for i, size in enumerate(dbscan['cluster_sizes']):
                        f.write(f"Cluster {i}: {size} samples, ")
                    f.write("\n\n")
                
                if 'PCA' in self.modeling_results['models']:
                    pca = self.modeling_results['models']['PCA']
                    f.write("PCA Dimensionality Reduction:\n")
                    f.write(f"  Total Explained Variance (2D): {pca['total_explained_variance']:.4f}\n")
                    f.write(f"  PC1 Explained Variance: {pca['explained_variance_ratio'][0]:.4f}\n")
                    f.write(f"  PC2 Explained Variance: {pca['explained_variance_ratio'][1]:.4f}\n\n")
            
            # Feature Importance
            if 'feature_importance' in self.modeling_results and self.modeling_results['feature_importance']:
                f.write("FEATURE IMPORTANCE\n")
                f.write("-" * 80 + "\n\n")
                
                best_model = self.modeling_results['best_model']
                if best_model and best_model in self.modeling_results['feature_importance']:
                    f.write(f"Feature Importance from {best_model}:\n\n")
                    
                    importances = self.modeling_results['feature_importance'][best_model]
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                    
                    for feature, importance in sorted_features:
                        f.write(f"{feature:<30} {importance:.4f}\n")
                    
                    f.write("\n")
            
            # Visualization References
            f.write("VISUALIZATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("The following visualizations have been generated in the Modeling folder:\n\n")
            
            # List generated plots
            modeling_dir = self.output_dir / "Modeling"
            for file in modeling_dir.glob(f"*{self.timestamp}*.png"):
                f.write(f"  - {file.name}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logging.info(f"Modeling report generated at {report_path}")
        
    def _generate_executive_insights(self):
        """Generate executive insights from the data using optimized analysis"""
        insights = []
        
        # Data profile insight
        insights.append(f"The dataset contains {len(self.processed_data) if self.processed_data is not None else 'N/A'} records with {len(self.processed_data.columns) if self.processed_data is not None else 'N/A'} variables")
        
        # Categorical variable insights
        for col in self.categorical_cols[:3]:  # Limit to top 3 for performance
            if col in self.processed_data.columns:
                # Use efficient value_counts with vectorized operations
                top_value = self.processed_data[col].value_counts(normalize=True).head(1)
                if not top_value.empty:
                    category = top_value.index[0]
                    percentage = top_value.iloc[0] * 100
                    insights.append(f"Most common {col}: '{category}' ({percentage:.2f}%)")
        
        # Missing data insight if applicable
        if hasattr(self, 'data_quality_metrics'):
            metrics = self.data_quality_metrics
            if 'missing_count' in metrics and metrics['missing_count'] > 0:
                insights.append(f"Data contains {metrics['missing_count']} missing values ({metrics['missing_pct']:.2f}%)")
        
        # Numeric variable insights (if any)
        if self.numeric_cols:
            for col in self.numeric_cols[:2]:  # Limit to top 2 for performance
                if col in self.processed_data.columns:
                    mean_val = self.processed_data[col].mean()
                    insights.append(f"Average {col}: {mean_val:.2f}")
        else:
            insights.append("No numeric columns available for statistical modeling")
        
        # Time series insight (if applicable)
        if self.datetime_cols:
            date_col = self.datetime_cols[0]
            min_date = self.processed_data[date_col].min()
            max_date = self.processed_data[date_col].max()
            time_span = (max_date - min_date).days
            insights.append(f"Data spans {time_span} days from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Add recommendation
        insights.append("Recommendation: Focus on understanding categorical relationships as the dataset lacks numerical variables")
        
        return insights
        
    def generate_comprehensive_report(self):
        """Generate a comprehensive report summarizing all analyses"""
        # Create output directory for reports if it doesn't exist
        report_dir = self.output_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate executive insights
        executive_insights = self._generate_executive_insights()
        
        # Path for comprehensive report
        report_path = report_dir / f"comprehensive_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            # Header
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE DATA ANALYSIS AND BUSINESS INTELLIGENCE REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            # Metadata
            f.write("REPORT METADATA\n")
            f.write("-" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {os.path.basename(self.data_path)}\n")
            f.write(f"Records: {len(self.processed_data) if self.processed_data is not None else 'N/A'}\n")
            f.write(f"Variables: {len(self.processed_data.columns) if self.processed_data is not None else 'N/A'}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write("This report provides a comprehensive analysis of sales data, including detailed market insights,\n")
            f.write("trend analysis, and actionable recommendations for business optimization.\n\n")
            for insight in executive_insights:
                f.write(f"• {insight}\n")
            f.write("\n")
            
            # Table of Contents
            f.write("TABLE OF CONTENTS\n")
            f.write("-" * 50 + "\n")
            f.write("1. Market Analysis and Key Findings\n")
            f.write("2. Data Quality Assessment\n")
            f.write("3. Sales Pattern Analysis\n")
            f.write("4. Geographic Distribution Analysis\n")
            f.write("5. Product Performance Analysis\n")
            f.write("6. Sales Channel Analysis\n")
            f.write("7. Price Analysis and Optimization\n")
            f.write("8. Forecasting and Trends\n")
            f.write("9. Strategic Recommendations\n")
            f.write("10. Technical Appendix\n\n")
            
            # 1. Market Analysis and Key Findings
            f.write("1. MARKET ANALYSIS AND KEY FINDINGS\n")
            f.write("=" * 50 + "\n")
            if self.processed_data is not None:
                # Sales Distribution
                if 'SALE COUNTRY' in self.categorical_cols:
                    country_dist = self.processed_data['SALE COUNTRY'].value_counts()
                    f.write("\nA. Geographic Market Distribution:\n")
                    f.write("-" * 30 + "\n")
                    for country, count in country_dist.items():
                        percentage = (count/len(self.processed_data))*100
                        f.write(f"• {country}: {count} sales ({percentage:.2f}%)\n")
                
                # Product Distribution
                if 'PART NAME' in self.categorical_cols:
                    product_dist = self.processed_data['PART NAME'].value_counts()
                    f.write("\nB. Product Category Analysis:\n")
                    f.write("-" * 30 + "\n")
                    for product, count in product_dist.items():
                        percentage = (count/len(self.processed_data))*100
                        f.write(f"• {product}: {count} units ({percentage:.2f}%)\n")
                
                # Sales Method Analysis
                if 'SALE METHOD' in self.categorical_cols:
                    method_dist = self.processed_data['SALE METHOD'].value_counts()
                    f.write("\nC. Sales Channel Performance:\n")
                    f.write("-" * 30 + "\n")
                    for method, count in method_dist.items():
                        percentage = (count/len(self.processed_data))*100
                        f.write(f"• {method}: {count} transactions ({percentage:.2f}%)\n")
            
            # 2. Data Quality Assessment
            f.write("\n2. DATA QUALITY ASSESSMENT\n")
            f.write("=" * 50 + "\n")
            if hasattr(self, 'data_quality_metrics'):
                metrics = self.data_quality_metrics
                f.write("\nA. Data Integrity Metrics:\n")
                f.write("-" * 30 + "\n")
                if 'duplicate_count' in metrics:
                    f.write(f"• Duplicate Records: {metrics['duplicate_count']} identified and removed\n")
                    f.write(f"  - Impact: {metrics['duplicate_pct']:.2f}% of original dataset\n")
                if 'missing_count' in metrics:
                    f.write(f"• Missing Values: {metrics['missing_count']} total\n")
                    f.write(f"  - Impact: {metrics['missing_pct']:.2f}% of total data points\n")
            
            # 3. Sales Pattern Analysis
            f.write("\n3. SALES PATTERN ANALYSIS\n")
            f.write("=" * 50 + "\n")
            if self.processed_data is not None:
                # Cross-tabulation analysis
                if 'SALE COUNTRY' in self.categorical_cols and 'PART NAME' in self.categorical_cols:
                    f.write("\nA. Product Performance by Market:\n")
                    f.write("-" * 30 + "\n")
                    cross_tab = pd.crosstab(self.processed_data['SALE COUNTRY'], 
                                          self.processed_data['PART NAME'], 
                                          normalize='index') * 100
                    for country in cross_tab.index:
                        f.write(f"\n• {country} Market:\n")
                        for product in cross_tab.columns:
                            f.write(f"  - {product}: {cross_tab.loc[country, product]:.2f}% of market share\n")
            
            # 4. Geographic Distribution Analysis
            f.write("\n4. GEOGRAPHIC DISTRIBUTION ANALYSIS\n")
            f.write("=" * 50 + "\n")
            if 'SALE COUNTRY' in self.categorical_cols:
                country_analysis = self.processed_data['SALE COUNTRY'].value_counts()
                f.write("\nA. Market Penetration Analysis:\n")
                f.write("-" * 30 + "\n")
                total_sales = len(self.processed_data)
                for country, sales in country_analysis.items():
                    penetration = (sales/total_sales)*100
                    f.write(f"• {country}:\n")
                    f.write(f"  - Sales Volume: {sales} units\n")
                    f.write(f"  - Market Share: {penetration:.2f}%\n")
            
            # 5. Product Performance Analysis
            f.write("\n5. PRODUCT PERFORMANCE ANALYSIS\n")
            f.write("=" * 50 + "\n")
            if 'PART NAME' in self.categorical_cols:
                product_analysis = self.processed_data['PART NAME'].value_counts()
                f.write("\nA. Product Demand Analysis:\n")
                f.write("-" * 30 + "\n")
                for product, demand in product_analysis.items():
                    share = (demand/total_sales)*100
                    f.write(f"• {product}:\n")
                    f.write(f"  - Total Sales: {demand} units\n")
                    f.write(f"  - Product Mix Share: {share:.2f}%\n")
            
            # 6. Sales Channel Analysis
            f.write("\n6. SALES CHANNEL ANALYSIS\n")
            f.write("=" * 50 + "\n")
            if 'SALE METHOD' in self.categorical_cols:
                channel_analysis = self.processed_data['SALE METHOD'].value_counts()
                f.write("\nA. Channel Effectiveness:\n")
                f.write("-" * 30 + "\n")
                for channel, volume in channel_analysis.items():
                    effectiveness = (volume/total_sales)*100
                    f.write(f"• {channel}:\n")
                    f.write(f"  - Transaction Volume: {volume}\n")
                    f.write(f"  - Channel Share: {effectiveness:.2f}%\n")
            
            # 7. Price Analysis
            f.write("\n7. PRICE ANALYSIS AND OPTIMIZATION\n")
            f.write("=" * 50 + "\n")
            if 'PRICE' in self.categorical_cols:
                price_data = self.processed_data['PRICE'].value_counts()
                f.write("\nA. Price Point Distribution:\n")
                f.write("-" * 30 + "\n")
                for price, frequency in price_data.head(10).items():
                    f.write(f"• {price}: {frequency} transactions\n")
            
            # 8. Forecasting and Trends
            f.write("\n8. FORECASTING AND TRENDS\n")
            f.write("=" * 50 + "\n")
            f.write("\nA. Demand Forecast by Product Category:\n")
            f.write("-" * 30 + "\n")
            if 'PART NAME' in self.categorical_cols:
                for product in self.processed_data['PART NAME'].unique():
                    product_count = len(self.processed_data[self.processed_data['PART NAME'] == product])
                    avg_daily_demand = product_count / 30  # Assuming 30-day period
                    f.write(f"\n• {product}:\n")
                    f.write(f"  - Current Monthly Demand: {product_count} units\n")
                    f.write(f"  - Average Daily Demand: {avg_daily_demand:.2f} units\n")
                    f.write(f"  - Projected Monthly Demand: {int(product_count * 1.1)} units (10% growth assumption)\n")
            
            # 9. Strategic Recommendations
            f.write("\n9. STRATEGIC RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n")
            
            # Market Expansion Recommendations
            f.write("\nA. Market Expansion Opportunities:\n")
            f.write("-" * 30 + "\n")
            if 'SALE COUNTRY' in self.categorical_cols:
                low_penetration_markets = country_analysis[country_analysis < country_analysis.mean()]
                f.write("• Target Markets for Growth:\n")
                for market in low_penetration_markets.index:
                    f.write(f"  - {market}: Potential for {(country_analysis.mean() - low_penetration_markets[market]):.0f} additional units\n")
            
            # Product Mix Optimization
            f.write("\nB. Product Portfolio Optimization:\n")
            f.write("-" * 30 + "\n")
            if 'PART NAME' in self.categorical_cols:
                f.write("• High-Potential Products:\n")
                for product, demand in product_analysis.head(3).items():
                    f.write(f"  - {product}: Increase production capacity by 20% to meet demand\n")
            
            # Sales Channel Enhancement
            f.write("\nC. Sales Channel Optimization:\n")
            f.write("-" * 30 + "\n")
            if 'SALE METHOD' in self.categorical_cols:
                f.write("• Channel Strategy Recommendations:\n")
                for channel, volume in channel_analysis.items():
                    if volume < channel_analysis.mean():
                        f.write(f"  - {channel}: Implement promotional campaign to boost sales\n")
            
            # 10. Technical Appendix
            f.write("\n10. TECHNICAL APPENDIX\n")
            f.write("=" * 50 + "\n")
            f.write("\nA. Data Processing Summary:\n")
            f.write("-" * 30 + "\n")
            f.write("• Data Cleaning Steps:\n")
            if hasattr(self, 'data_quality_metrics'):
                for step, metric in self.data_quality_metrics.items():
                    f.write(f"  - {step}: {metric}\n")
            
            f.write("\nB. Statistical Methods Used:\n")
            f.write("-" * 30 + "\n")
            f.write("• Descriptive Statistics\n")
            f.write("• Frequency Analysis\n")
            f.write("• Cross-tabulation Analysis\n")
            f.write("• Time Series Analysis (where applicable)\n")
            
            # End of Report
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")
        
        logging.info(f"Enhanced comprehensive report generated at {report_path}")
        
        # Generate HTML report with visualizations
        html_report_path = report_dir / f"analysis_report.html"
        visualization_files = []
        
        # Find all visualization files
        categories = ["DataPreprocessing", "EDA", "TimeSeries", "Modeling"]
        for category in categories:
            category_dir = self.output_dir / category
            if category_dir.exists():
                viz_files = list(category_dir.glob(f"*{self.timestamp}*.png"))
                visualization_files.extend(viz_files)
        
        # Add visualization files that may be directly in the output directory
        direct_viz_files = list(self.output_dir.glob(f"*{self.timestamp}*.png"))
        visualization_files.extend(direct_viz_files)
        
        # Generate HTML report
        self._generate_html_report(visualization_files, html_report_path)
        
        return report_path

    def run_all(self):
        """Run the complete analysis pipeline with optimized performance"""
        start_time = time.time()
        logging.info("Starting complete analysis pipeline...")
        
        # 1. Load data
        if not self.load_data():
            logging.error("Failed to load data. Analysis aborted.")
            return False
        
        # 2. Preprocess data
        if not self.preprocess_data():
            logging.error("Failed to preprocess data. Analysis aborted.")
            return False
        
        # 3. Create thread pool for parallel tasks
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit EDA task
            eda_future = executor.submit(self.perform_eda)
            
            # Submit time series analysis task if applicable
            ts_future = None
            if self.datetime_cols:
                try:
                    ts_future = executor.submit(self.perform_time_series_analysis)
                except Exception as e:
                    logging.error(f"Error during time series analysis: {str(e)}")
                    logging.error("Continuing with other analysis steps...")
            
            # Wait for EDA to complete
            try:
                eda_future.result()
            except Exception as e:
                logging.error(f"Error during EDA: {str(e)}")
            
            # Wait for time series analysis to complete
            if ts_future:
                try:
                    ts_future.result()
                except Exception as e:
                    logging.error(f"Error during time series analysis: {str(e)}")
        
        # 4. Perform predictive modeling
        try:
            self.perform_predictive_modeling()
        except Exception as e:
            logging.error(f"Error during predictive modeling: {str(e)}")
            logging.error("Continuing with report generation...")
        
        # 5. Generate comprehensive report
        try:
            report_path = self.generate_comprehensive_report()
        except Exception as e:
            logging.error(f"Error generating comprehensive report: {str(e)}")
            report_path = None
        
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Complete analysis pipeline finished in {execution_time:.2f} seconds")
        
        if report_path:
            logging.info(f"Comprehensive report saved to {report_path}")
            return True
        else:
            logging.error("Analysis completed with errors. Check logs for details.")
            return False

    def _generate_html_report(self, visualization_files, output_path):
        """Generate an HTML report with embedded visualizations using optimized code"""
        # Start with base HTML structure
        html = ["<!DOCTYPE html>", "<html lang='en'>", "<head>", 
                "<meta charset='UTF-8'>",
                "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
                "<title>Data Analysis Report</title>",
                "<style>"]
        
        # Add CSS styles - optimized for rendering performance
        css = """
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
            h3 { color: #3498db; margin-top: 20px; }
            .viz-container { display: flex; flex-wrap: wrap; justify-content: space-between; }
            .viz-item { margin: 10px 0; width: 100%; }
            .viz-item img { max-width: 100%; height: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .summary-box { background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }
            .metric-group { display: flex; flex-wrap: wrap; }
            .metric { background: #f8f9fa; border-radius: 4px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
            .metric h4 { margin: 0 0 10px 0; color: #2c3e50; }
            .metric p { margin: 0; font-size: 1.2em; font-weight: bold; color: #3498db; }
            .nav-tabs { display: flex; list-style: none; padding: 0; margin-bottom: 0; }
            .nav-tabs li { padding: 10px 15px; background: #f8f9fa; border: 1px solid #dee2e6; border-bottom: none; cursor: pointer; }
            .nav-tabs li.active { background: white; border-bottom: 1px solid white; }
            .tab-content { border: 1px solid #dee2e6; padding: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        """
        html.append(css)
        html.extend(["</style>", "</head>", "<body>"])
        
        # Add report header
        html.append("<h1>Comprehensive Data Analysis Report</h1>")
        
        # Add timestamp and dataset info
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p>Dataset: {os.path.basename(self.data_path)}</p>")
        
        # Add executive summary section
        html.append("<div class='summary-box'>")
        html.append("<h2>Executive Summary</h2>")
        html.append("<ul>")
        
        # Add insights
        executive_insights = self._generate_executive_insights()
        for insight in executive_insights:
            html.append(f"<li>{insight}</li>")
        
        html.append("</ul>")
        html.append("</div>")
        
        # Add key metrics section
        html.append("<h2>Dataset Overview</h2>")
        html.append("<div class='metric-group'>")
        
        # Records count
        html.append("<div class='metric'>")
        html.append("<h4>Records</h4>")
        html.append(f"<p>{len(self.processed_data) if self.processed_data is not None else 'N/A'}</p>")
        html.append("</div>")
        
        # Variables count
        html.append("<div class='metric'>")
        html.append("<h4>Variables</h4>")
        html.append(f"<p>{len(self.processed_data.columns) if self.processed_data is not None else 'N/A'}</p>")
        html.append("</div>")
        
        # Numeric variables count
        html.append("<div class='metric'>")
        html.append("<h4>Numeric Variables</h4>")
        html.append(f"<p>{len(self.numeric_cols)}</p>")
        html.append("</div>")
        
        # Categorical variables count
        html.append("<div class='metric'>")
        html.append("<h4>Categorical Variables</h4>")
        html.append(f"<p>{len(self.categorical_cols)}</p>")
        html.append("</div>")
        
        html.append("</div>")  # End metric-group
        
        # Additional debug info
        logging.info(f"Found {len(visualization_files)} visualization files for HTML report")
        for viz_file in visualization_files:
            logging.info(f"Visualization file: {viz_file}")
        
        # Check for visualization files directly in the output directory
        direct_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        if direct_files:
            logging.info(f"Found {len(direct_files)} direct PNG files in output dir: {', '.join(direct_files)}")
            # Add these files to visualization_files if not already present
            for file in direct_files:
                file_path = self.output_dir / file
                if file_path not in visualization_files:
                    visualization_files.append(file_path)
        
        # Add visualizations section
        html.append("<h2>Visualizations</h2>")
        
        # Organize visualizations by type
        viz_categories = self._organize_visualizations(visualization_files)
        
        # If there are no organized visualizations, create a default category
        if not viz_categories:
            viz_categories = {'All Visualizations': []}
            # Try to find PNG files directly
            for file in os.listdir(self.output_dir):
                if file.endswith('.png'):
                    viz_categories['All Visualizations'].append(self.output_dir / file)
        
        # Add tabs for visualization categories
        html.append("<ul class='nav-tabs'>")
        
        # Create a tab for each visualization category
        for i, category in enumerate(viz_categories.keys()):
            active_class = "active" if i == 0 else ""
            html.append(f"<li class='{active_class}' onclick='showTab(\"{category}\")'>{category}</li>")
        
        html.append("</ul>")
        
        # Add tab content for each visualization category
        for i, (category, files) in enumerate(viz_categories.items()):
            display = "block" if i == 0 else "none"
            html.append(f"<div id='{category}' class='tab-content' style='display: {display};'>")
            
            if files:
                html.append("<div class='viz-container'>")
                for file_path in files:
                    # Extract just the filename for display
                    file_name = os.path.basename(str(file_path))
                    
                    # Get a readable title from the filename
                    readable_title = file_name.replace('_', ' ').replace('.png', '').title()
                    if hasattr(self, '_get_readable_title'):
                        try:
                            readable_title = self._get_readable_title(file_name)
                        except:
                            pass
                    
                    html.append("<div class='viz-item'>")
                    html.append(f"<h3>{readable_title}</h3>")
                    
                    # Try to embed the image using base64
                    try:
                        with open(file_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            html.append(f"<img src='data:image/png;base64,{img_data}' alt='{file_name}'>")
                    except Exception as e:
                        logging.warning(f"Error embedding visualization {file_path}: {str(e)}")
                        # Fallback: use direct file reference
                        try:
                            # Calculate relative path from HTML file to image
                            rel_path = os.path.relpath(str(file_path), os.path.dirname(str(output_path)))
                            html.append(f"<img src='{rel_path}' alt='{file_name}'>")
                            logging.info(f"Using relative path reference for {file_name}: {rel_path}")
                        except Exception as e2:
                            logging.error(f"Error creating relative path for {file_path}: {str(e2)}")
                            # Last resort: absolute path
                            html.append(f"<img src='{file_path}' alt='{file_name}'>")
                    
                    html.append("</div>")
                
                html.append("</div>")
            else:
                html.append("<p>No visualizations available for this category.</p>")
            
            html.append("</div>")
        
        # Add JavaScript for tabs
        html.append("<script>")
        html.append("function showTab(tabId) {")
        html.append("  const tabs = document.getElementsByClassName('tab-content');")
        html.append("  for (let i = 0; i < tabs.length; i++) {")
        html.append("    tabs[i].style.display = 'none';")
        html.append("  }")
        html.append("  document.getElementById(tabId).style.display = 'block';")
        html.append("  const navTabs = document.getElementsByClassName('nav-tabs')[0].getElementsByTagName('li');")
        html.append("  for (let i = 0; i < navTabs.length; i++) {")
        html.append("    navTabs[i].className = '';")
        html.append("  }")
        html.append("  const activeTab = Array.from(navTabs).find(tab => tab.textContent === tabId);")
        html.append("  if (activeTab) activeTab.className = 'active';")
        html.append("}")
        html.append("</script>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        # Write the HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
        
        logging.info(f"HTML report with visualizations generated at {output_path}")
        
    def _organize_visualizations(self, visualization_files):
        """Organize visualization files by category for HTML report - optimized for performance"""
        viz_categories = {
            'Distribution Analysis': [],
            'Categorical Analysis': [],
            'Relationship Analysis': [],
            'Other Visualizations': []
        }
        
        # Use string operations instead of regex for better performance
        for file_path in visualization_files:
            file_name = str(file_path.name).lower()
            
            if any(x in file_name for x in ['distribution', 'pie_chart', 'histogram']):
                viz_categories['Distribution Analysis'].append(file_path)
            elif any(x in file_name for x in ['barplot', 'bar_chart']):
                viz_categories['Categorical Analysis'].append(file_path)
            elif any(x in file_name for x in ['heatmap', 'correlation', 'vs_']):
                viz_categories['Relationship Analysis'].append(file_path)
            else:
                viz_categories['Other Visualizations'].append(file_path)
        
        # Remove empty categories
        return {k: v for k, v in viz_categories.items() if v}

    def _get_readable_title(self, filename):
        """Convert a filename to a readable title"""
        # Remove timestamp pattern
        readable = filename.replace(self.timestamp, '')
        # Remove file extension
        readable = os.path.splitext(readable)[0]
        # Replace underscores with spaces
        readable = readable.replace('_', ' ')
        # Remove leading/trailing spaces
        readable = readable.strip()
        # Capitalize words
        return readable.title()

def main():
    """Main function to run the data analysis"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Data Analysis Tool')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output', '-o', type=str, default='OUTPUT', help='Output directory for reports and visuals')
    parser.add_argument('--missing-threshold', '-mt', type=float, default=0.3, 
                        help='Threshold for dropping columns with missing values (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Initialize the analyzer
    analyzer = EnhancedDataAnalyzer(
        data_path=args.input,
        output_dir=args.output,
        missing_threshold=args.missing_threshold
    )
    
    # Run the complete analysis pipeline
    success = analyzer.run_all()
    
    if success:
        print(f"\nAnalysis completed successfully!")
        print(f"Reports and visualizations saved to: {analyzer.output_dir}")
    else:
        print(f"\nAnalysis completed with errors. Please check the logs.")

if __name__ == "__main__":
    main() 