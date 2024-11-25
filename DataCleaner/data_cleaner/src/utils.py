import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.
    """
    return pd.read_csv(file_path)

def save_csv(df: pd.DataFrame, file_path: str):
    """
    Saves a DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)
