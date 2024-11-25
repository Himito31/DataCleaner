import matplotlib.pyplot as plt

def plot_missing_values(df):
    """
    Visualizes missing values in the DataFrame.
    """
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    missing_counts.plot(kind="bar", color="skyblue")
    plt.title("Missing Values by Column")
    plt.ylabel("Count")
    plt.show()
