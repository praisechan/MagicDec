#!/usr/bin/env python3
"""
Aggregate results from hyperparameter sweep into a single CSV file.

This script reads accumulated_log.csv from each hyperparameter sweep directory
and combines them into a single CSV file, optionally taking just the final row
from each run.
"""

import os
import pandas as pd
from pathlib import Path
import argparse


def aggregate_sweep_results(logs_dir, output_csv, take_final_only=True):
    """
    Aggregate sweep results from multiple directories into a single CSV.
    
    Args:
        logs_dir: Path to the root sweep logs directory
        output_csv: Path to output CSV file
        take_final_only: If True, take only the final row from each run; 
                        if False, take all rows from all runs
    """
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        raise ValueError(f"Logs directory not found: {logs_dir}")
    
    # Get all subdirectories (each represents a hyperparameter combination)
    result_dirs = sorted([d for d in logs_path.iterdir() if d.is_dir()])
    
    if not result_dirs:
        raise ValueError(f"No result directories found in {logs_dir}")
    
    all_rows = []
    
    for result_dir in result_dirs:
        accumulated_log = result_dir / "accumulated_log.csv"
        
        if not accumulated_log.exists():
            print(f"Warning: accumulated_log.csv not found in {result_dir.name}")
            continue
        
        try:
            df = pd.read_csv(accumulated_log)
            
            if take_final_only:
                # Take only the final row
                final_row = df[df['step'] == 'final'].iloc[0:1]
                if not final_row.empty:
                    all_rows.append(final_row)
                else:
                    # If no 'final' row, take the last row
                    all_rows.append(df.iloc[-1:])
            else:
                # Take all rows
                all_rows.append(df)
                
        except Exception as e:
            print(f"Error reading {accumulated_log}: {e}")
            continue
    
    if not all_rows:
        raise ValueError("No results were successfully read from any directory")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_rows, ignore_index=True)
    
    # Reorder columns to put hyperparameters first
    hyperparams = ['gamma1', 'gamma2', 'budget1', 'budget2']
    other_cols = [col for col in combined_df.columns if col not in hyperparams]
    
    ordered_cols = hyperparams + other_cols
    combined_df = combined_df[ordered_cols]
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Aggregated results saved to {output_csv}")
    print(f"Total rows: {len(combined_df)}")
    print(f"\nFirst few rows:")
    print(combined_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate hyperparameter sweep results into a single CSV"
    )
    parser.add_argument(
        "--logs_dir",
        default="tests/RetrievalAttention_new/logs/sweep_static_hparams_8k",
        help="Path to the sweep logs directory"
    )
    parser.add_argument(
        "--output",
        default="tests/RetrievalAttention_new/aggregated_results.csv",
        help="Path to the output CSV file"
    )
    parser.add_argument(
        "--all_rows",
        action="store_true",
        help="Include all rows from each run (default: final row only)"
    )
    
    args = parser.parse_args()
    
    aggregate_sweep_results(
        args.logs_dir,
        args.output,
        take_final_only=not args.all_rows
    )
