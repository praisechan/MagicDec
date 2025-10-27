#!/usr/bin/env python3
"""
Script to reorganize CSV data by grouping plane_index values with same remainder when divided by 32.
Each row represents a unique plane (plane_index % 32), and columns represent different quotients.
"""

import pandas as pd
import numpy as np
import os

def reorganize_csv(input_file, output_file):
    """
    Reorganize CSV data by grouping plane indices with same remainder mod 32.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    print(f"Reading data from {input_file}")
    
    # Read the original CSV
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Plane index range: {df['plane_index'].min()} to {df['plane_index'].max()}")
    
    # Calculate plane group (remainder) and quotient
    df['plane_group'] = df['plane_index'] % 32
    df['quotient'] = df['plane_index'] // 32
    
    print(f"Number of unique plane groups: {df['plane_group'].nunique()}")
    print(f"Number of unique quotients: {df['quotient'].nunique()}")
    
    # Pivot the data: rows = plane_group, columns = reads_count for each quotient
    pivot_df = df.pivot(index='plane_group', columns='quotient', values='reads_count')
    
    # Rename columns to reads_count_X format
    pivot_df.columns = [f'reads_count_{col}' for col in pivot_df.columns]
    
    # Reset index to make plane_group a regular column
    pivot_df = pivot_df.reset_index()
    
    print(f"Reorganized data shape: {pivot_df.shape}")
    
    # Save to output file
    pivot_df.to_csv(output_file, index=False)
    print(f"Reorganized data saved to {output_file}")
    
    # Display summary statistics
    print("\nSummary of reorganized data:")
    print(f"- Number of planes (rows): {len(pivot_df)}")
    print(f"- Number of quotient columns: {len(pivot_df.columns) - 1}")
    print(f"- Column names: plane_group, {', '.join(pivot_df.columns[1:6])}...{pivot_df.columns[-1]}")
    
    return pivot_df

def main():
    # File paths
    input_file = "data/reads_per_head_raw_data_Meta-Llama-3.1-8B_32800_budget0.10_replica4.csv"
    output_file = "data/reads_per_head_reorganized_Meta-Llama-3.1-8B_32800_budget0.10_replica4.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Make sure you're running this script from the correct directory.")
        return
    
    try:
        # Reorganize the data
        reorganized_df = reorganize_csv(input_file, output_file)
        
        # Show a preview of the reorganized data
        print("\nPreview of reorganized data:")
        print(reorganized_df.head())
        
        print("\nExample: Plane group 3 data:")
        plane_3_data = reorganized_df[reorganized_df['plane_group'] == 3]
        if not plane_3_data.empty:
            # Show first few columns for plane group 3
            cols_to_show = ['plane_group'] + [col for col in reorganized_df.columns if col.startswith('reads_count_')][:5]
            print(plane_3_data[cols_to_show].to_string(index=False))
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return

if __name__ == "__main__":
    main()