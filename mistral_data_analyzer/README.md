# Mistral Data Analyzer

A completely offline, local Excel data analysis system powered by Mistral 7B. This tool analyzes Excel data and generates insights and visualizations without requiring internet connectivity.

## Features

- Load and analyze Excel files (.xlsx, .xls)
- Generate statistical summaries and insights using the Mistral 7B LLM model
- Create visualizations (heatmaps, histograms, bar charts, time series)
- Export analysis as text report with references to visualizations
- **Auto-refresh monitoring** to automatically analyze Excel files when they change
- Works completely offline once set up

## Requirements

- Python 3.8+
- Ollama (for running Mistral 7B locally)
- 8GB+ GPU for optimal performance

## Setup Instructions

1. **Install Ollama**
   
   Download and install Ollama from: https://ollama.com/download/windows
   
2. **Download the Mistral 7B model**
   
   After installing Ollama, open a terminal/PowerShell and run:
   ```
   ollama pull mistral:7b-instruct-v0.2
   ```

3. **Install Python dependencies**
   
   In the project directory, run:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Single Analysis Mode

1. **Make sure Ollama is running**
   
   Start Ollama (it should be running as a service in Windows).

2. **Run the analyzer**
   
   Basic usage:
   ```
   python main.py --excel your_data.xlsx
   ```
   
   Advanced options:
   ```
   python main.py --excel your_data.xlsx --model mistral:7b-instruct-v0.2 --output custom_report.txt
   ```

3. **View results**
   
   - Text report will be saved to `analysis_report.txt` (or your custom output file)
   - Visualizations will be saved to the `generated_plots/` directory

### Auto-Refresh Mode (recommended)

1. **Make sure Ollama is running**

2. **Place Excel files in the INPUT directory**
   
   Default location: `C:\Users\DOGEBABA\Desktop\DOGEAI\INPUT`

3. **Start the auto-analyzer**
   ```
   python auto_analyzer.py
   ```

4. **How it works**
   - The program watches the INPUT directory for new or modified Excel files
   - When a file is added or changed, it automatically runs the analysis
   - Each analysis creates a timestamped folder in the OUTPUT directory
   - Press Ctrl+C to stop the auto-analyzer

## Command Line Options

### Main Analyzer
- `--excel, -e` (Required): Path to Excel file to analyze
- `--model, -m` (Optional): Mistral model to use (default: mistral:7b-instruct-v0.2)
- `--output, -o` (Optional): Output file for analysis report (default: analysis_report.txt)
- `--ollama-url` (Optional): URL for Ollama API (default: http://localhost:11434)

### Auto Analyzer
- `--input-dir` (Optional): Directory to watch for Excel files (default: C:\Users\DOGEBABA\Desktop\DOGEAI\INPUT)
- `--model, -m` (Optional): Mistral model to use (default: mistral:7b-instruct-v0.2)
- `--ollama-url` (Optional): URL for Ollama API (default: http://localhost:11434)

## Example

```
# Single file analysis
python main.py --excel sales_data.xlsx --output sales_analysis.txt

# Auto-refresh mode
python auto_analyzer.py
```

## Troubleshooting

- If you get connection errors, make sure Ollama is running properly
- For visualization errors, check that all required Python dependencies are installed
- If the model is slow, try using a more efficient quantized version like `mistral:7b-instruct-v0.2-q4_K_S` 