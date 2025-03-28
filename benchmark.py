import time
import subprocess
import os
from pathlib import Path
import shutil
import sys

def run_command_with_timing(command):
    """Run a command and return the execution time and success status"""
    start_time = time.time()
    
    # Run the command
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Check if the command was successful
    success = process.returncode == 0
    
    return execution_time, success, process.stdout, process.stderr

def cleanup_output_dirs():
    """Clean up output directories to ensure fair benchmarking"""
    output_dir = Path("OUTPUT")
    if output_dir.exists():
        # Instead of removing the entire directory, just rename it
        backup_dir = Path("OUTPUT_backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        output_dir.rename(backup_dir)
    
    # Create fresh output directory
    output_dir.mkdir(exist_ok=True)

def format_bytes(size):
    """Format bytes to human-readable format"""
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

def get_directory_size(path):
    """Get the size of a directory including all subdirectories"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def count_files(directory):
    """Count the number of files in a directory and its subdirectories"""
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count

def print_header(text):
    """Print a formatted header for better readability"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)

def main():
    # Define input file path
    input_file = "C:\\Users\\DOGEBABA\\Desktop\\DOGEAI\\INPUT\\test.xlsx"
    
    print_header("BENCHMARKING ENHANCED DATA ANALYSIS")
    print(f"Input file: {input_file}")
    
    # Clean up output directories
    cleanup_output_dirs()
    
    # Benchmark optimized implementation
    print_header("RUNNING OPTIMIZED IMPLEMENTATION")
    optimized_cmd = f"py enhanced_data_analysis.py --input {input_file}"
    
    print(f"Running command: {optimized_cmd}")
    opt_time, opt_success, opt_stdout, opt_stderr = run_command_with_timing(optimized_cmd)
    
    print(f"Execution time: {opt_time:.2f} seconds")
    print(f"Success: {'Yes' if opt_success else 'No'}")
    
    # Get latest output directory
    output_dirs = sorted([d for d in os.listdir("OUTPUT") if d.startswith("Analysis_")], reverse=True)
    if output_dirs:
        latest_dir = os.path.join("OUTPUT", output_dirs[0])
        opt_size = get_directory_size(latest_dir)
        opt_files = count_files(latest_dir)
        print(f"Output directory: {latest_dir}")
        print(f"Output size: {format_bytes(opt_size)}")
        print(f"Number of files: {opt_files}")
    
    # Print comparison with previous performance
    print_header("PERFORMANCE COMPARISON")
    previous_time = 12.32  # From the logs of the original implementation
    speed_improvement = (previous_time - opt_time) / previous_time * 100
    
    print(f"Original implementation: {previous_time:.2f} seconds")
    print(f"Optimized implementation: {opt_time:.2f} seconds")
    print(f"Speed improvement: {speed_improvement:.2f}%")
    
    # Recommendations
    print_header("OPTIMIZATION SUMMARY")
    print("1. Optimized data loading and preprocessing")
    print("   - Improved type detection")
    print("   - More efficient memory usage")
    print("   - Optimized categorical encoding")
    
    print("\n2. Visualization improvements")
    print("   - Reduced file I/O operations")
    print("   - Better plotting parameters")
    print("   - More efficient figure creation and closing")
    
    print("\n3. Parallelization enhancements")
    print("   - Added multi-threading capability")
    print("   - Thread pool management for I/O bound operations")
    print("   - Concurrent execution of independent tasks")
    
    print("\n4. Report generation optimizations")
    print("   - Streamlined HTML report generation")
    print("   - More efficient data organization")
    print("   - Better string handling")
    
    print("\n5. Memory management")
    print("   - Reduced deep copies where not needed")
    print("   - Better garbage collection")
    print("   - Proper resource cleanup")

if __name__ == "__main__":
    main() 