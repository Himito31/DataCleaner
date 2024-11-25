import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataCleaner object.
        :param df: pandas DataFrame to be processed.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df
        self.log = []

    def handle_missing(self, strategy: str = "mean", fill_value=None):
        """
        Handles missing values in the DataFrame.
        :param strategy: Filling strategy ('mean', 'median', 'mode', 'constant').
        :param fill_value: Value for the 'constant' strategy.
        """
        if strategy not in ["mean", "median", "mode", "constant"]:
            raise ValueError("Invalid strategy.")
        
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if strategy in ["mean", "median"] and not pd.api.types.is_numeric_dtype(self.df[column]):
                    # Skip non-numeric columns for mean/median
                    self.log.append(f"Skipped non-numeric column '{column}' for '{strategy}' strategy.")
                    continue
                if strategy == "mean":
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == "median":
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == "mode":
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                elif strategy == "constant":
                    self.df[column].fillna(fill_value, inplace=True)
                self.log.append(f"Handled missing values in column '{column}' using '{strategy}' strategy.")


    def remove_outliers(self, z_thresh: float = 3.0):
        """
        Removes outliers using the Z-score method.
        :param z_thresh: Z-score threshold.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < z_thresh]
            self.log.append(f"Removed outliers in column '{col}' with Z-threshold={z_thresh}.")

    def normalize(self, method: str = "min-max"):
        """
        Normalizes numerical data.
        :param method: Normalization method ('min-max' or 'z-score').
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if method == "min-max":
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
            elif method == "z-score":
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            self.log.append(f"Normalized column '{col}' using '{method}' method.")

    def encode_categories(self, method: str = "one-hot"):
        """
        Transforms categorical variables.
        :param method: Encoding method ('one-hot' or 'label').
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if method == "one-hot":
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1).drop(columns=[col])
            elif method == "label":
                self.df[col] = self.df[col].astype("category").cat.codes
            self.log.append(f"Encoded column '{col}' using '{method}' method.")

    def save_log(self, path: str = "cleaning_log.txt"):
        """
        Saves the log of operations to a file.
        :param path: Path to the log file.
        """
        with open(path, "w") as f:
            f.write("\n".join(self.log))
