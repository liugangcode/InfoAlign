import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


class CSVToDataset:
    """
    A class to convert CSV files containing SMILES and property data to a standardized dataset format.
    
    The class extracts the SMILES column and specified property columns from a CSV file,
    renames the SMILES column, and saves the results to a specified location with metadata.
    """
    
    def __init__(self, task_name):
        """
        Initialize the converter with a task name.
        
        Args:
            task_name (str): Name of the task/dataset
        """
        self.task_name = task_name
        self.output_dir = Path(f"raw_data/{task_name}/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert(self, csv_path, smiles_column, property_columns, eval_metric):
        """
        Convert a CSV file to the standardized dataset format.
        
        Args:
            csv_path (str): Path to the input CSV file
            smiles_column (str): Name of the column containing SMILES strings
            property_columns (list): List of column names for properties to extract
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        if not os.path.exists(csv_path):
            print(f"Error: File {csv_path} does not exist.")
            return False
        df = pd.read_csv(csv_path)
            
        # Validate columns exist in the dataframe
        missing_columns = []
        if smiles_column not in df.columns:
            missing_columns.append(smiles_column)
            
        for col in property_columns:
            if col not in df.columns:
                missing_columns.append(col)
                
        if missing_columns:
            print(f"Error: The following columns are missing from the CSV: {', '.join(missing_columns)}")
            return False
            
        # Extract and rename columns
        selected_columns = [smiles_column] + property_columns
        result_df = df[selected_columns].copy()
        result_df.rename(columns={smiles_column: 'smiles'}, inplace=True)
        
        # Save the processed data
        output_path = self.output_dir / 'assays.csv.gz'
        result_df.to_csv(output_path, index=False, compression='gzip')
        
        # Create and save metadata
        self.num_tasks = len(property_columns)
        metadata = {
            'num_tasks': self.num_tasks,
            'start_column': 1,
            'eval_metric': eval_metric
        }
        
        with open(self.output_dir / 'meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Successfully processed {csv_path}")
        print(f"Saved data to {output_path}")
        print(f"Number of compounds: {len(result_df)}")
        print(f"Number of tasks: {self.num_tasks}")
        
        return True


# Example usage
if __name__ == "__main__":
    # Example:
    # converter = CSVToDataset("solubility")
    # converter.convert(
    #     csv_path="path/to/your/data.csv",
    #     smiles_column="Molecule",
    #     property_columns=["LogS", "Solubility_Class"]
    # )
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV files to standardized dataset format')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/dataset')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--smiles_column', type=str, required=True, help='Name of the column containing SMILES strings')
    parser.add_argument('--property_columns', type=str, nargs='+', required=True, 
                        help='Names of columns containing property values to extract')
    parser.add_argument('--eval_metric', type=str, default='roc_auc', help='Evaluation metric')

    args = parser.parse_args()
    assert args.eval_metric in ["roc_auc", "avg_mae"]
    
    converter = CSVToDataset(args.task_name)
    success = converter.convert(
        csv_path=args.csv_path,
        smiles_column=args.smiles_column,
        property_columns=args.property_columns,
        eval_metric=args.eval_metric
    )
    
    exit(0 if success else 1)
