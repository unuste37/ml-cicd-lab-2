##start
import pandas as pd
import os

if __name__ == "__main__":
    # SageMaker automatically downloads S3 input data to this local container path
    input_path = "/opt/ml/processing/input/churn.csv"
    
    # SageMaker expects processed data to be written to this specific output directory
    # Anything saved here is automatically uploaded back to S3 when the job finishes
    output_dir = "/opt/ml/processing/output"
    
    # We name the file 'train.csv' as it's the standard naming convention for SageMaker algorithms
    output_path = os.path.join(output_dir, "train.csv") 
    
    # Ensure the internal container directory exists before writing to it
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset downloaded from S3
    df = pd.read_csv(input_path)
    
    # Basic data cleaning: Remove any rows with missing values
    df = df.dropna()
    
    # SageMaker Built-in XGBoost requirements:
    # 1. The target variable (label) MUST be in the very first column.
    # 2. The CSV file must NOT have a header row.
    # 3. No index column should be included.
    df.to_csv(output_path, index=False, header=False) 
    
    print(f"✅ Data processed successfully and saved to: {output_path}")
##end
