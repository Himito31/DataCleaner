import pandas as pd
from data_cleaner.src.cleaner import DataCleaner
from data_cleaner.src.utils import save_csv

# Load sample data
df = pd.read_csv("reviews.csv")

# Initialize DataCleaner
cleaner = DataCleaner(df)

# Apply cleaning steps
cleaner.handle_missing(strategy="mean")  # Fills numeric columns only
cleaner.remove_outliers(z_thresh=3.0)
cleaner.normalize(method="z-score")
cleaner.encode_categories(method="one-hot")

# Save results
save_csv(cleaner.df, "cleaned_data.csv")
cleaner.save_log("cleaning_log.txt")

print("Data cleaning completed. Results saved in 'cleaned_data.csv'.")
