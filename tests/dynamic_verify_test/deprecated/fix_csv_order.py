import pandas as pd
import os

def fix_csv_column_order(csv_path):
    """
    Fix the column order in the CSV file to have 'experiment' first,
    followed by sorted range columns (0.00-0.10, 0.10-0.20, etc.)
    """
    # Read the existing CSV
    df = pd.read_csv(csv_path)
    
    # Identify range columns (all columns except 'experiment')
    range_columns = [col for col in df.columns if col != 'experiment']
    
    # Sort range columns by their starting value
    sorted_range_columns = sorted(range_columns, key=lambda x: float(x.split('-')[0]))
    
    # Define the desired column order: experiment first, then sorted ranges
    desired_columns = ['experiment'] + sorted_range_columns
    
    # Reorder the DataFrame
    df_sorted = df.reindex(columns=desired_columns, fill_value=0)
    
    # Create backup of original file
    backup_path = csv_path.replace('.csv', '_backup.csv')
    df.to_csv(backup_path, index=False)
    print(f"Backup created: {backup_path}")
    
    # Save the sorted DataFrame
    df_sorted.to_csv(csv_path, index=False)
    print(f"CSV file sorted and saved: {csv_path}")
    
    # Print the new column order
    print(f"New column order: {list(df_sorted.columns)}")
    
    return df_sorted

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = "/home/juchanlee/MagicDec/output/snapkv_Meta-Llama-3.1-8B_longbenchv1_histogram_data.csv"
    
    # Check if file exists
    if os.path.exists(csv_file_path):
        print(f"Processing file: {csv_file_path}")
        fixed_df = fix_csv_column_order(csv_file_path)
        print("Column order fixed successfully!")
        
        # Display first few rows to verify
        print("\nFirst 3 rows of the fixed CSV:")
        print(fixed_df.head(3).to_string())
        
    else:
        print(f"File not found: {csv_file_path}")
        print("Please check the file path.")