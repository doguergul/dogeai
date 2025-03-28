# Enhanced Data Analysis Tool

A powerful Python tool for comprehensive data analysis of Excel files, providing detailed business intelligence insights and visualizations.

## Features

- **Data Loading & Preprocessing**
  - Excel file import
  - Automatic data type detection
  - Handling of missing values and duplicates
  - Categorical variable encoding

- **Exploratory Data Analysis**
  - Descriptive statistics
  - Distribution analysis
  - Correlation analysis
  - Automated visualization generation

- **Business Intelligence Reports**
  - Comprehensive analysis report
  - Market analysis
  - Sales pattern analysis
  - Geographic distribution analysis
  - Product performance metrics
  - Sales channel effectiveness
  - Price analysis
  - Demand forecasting
  - Strategic recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/enhanced-data-analysis.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

```python
from enhanced_data_analysis import EnhancedDataAnalyzer

# Initialize analyzer
analyzer = EnhancedDataAnalyzer(data_path="your_data.xlsx")

# Run complete analysis
analyzer.run_all()
```

## Output Structure

The tool generates a structured output in the `OUTPUT` directory:
- Comprehensive text report
- Interactive HTML report with visualizations
- Detailed preprocessing reports
- EDA visualizations and statistics

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl

## License

MIT License - feel free to use and modify for your needs. 