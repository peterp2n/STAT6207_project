import pandas as pd





if __name__ == "__main__":

    csv_path = "/Users/timwong/Downloads/Repos/STAT6207_project/data/amazon.csv"
    destination_path = "/Users/timwong/Downloads/Repos/STAT6207_project/data/amazon_cleaned.csv"

    df = pd.read_csv(csv_path)
    dropped = df.dropna(subset=["isbn_13"]).drop_duplicates(subset=["isbn_13"])
    print(f"Dropped {len(df) - len(dropped)} rows with missing values.")
    dropped.to_csv(destination_path, index=False)