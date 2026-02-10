import pandas as pd
import os
from config import DATA_DIR

def load_data(filename="synthetic_control_data.csv"):
    csv_path = os.path.join(DATA_DIR, filename)
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run generate_dataset.py first.")
        return None
