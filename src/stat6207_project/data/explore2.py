import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":

    data_folder = Path("data")
    df = pd.read_csv(data_folder / "target_books_cleaned_v2.csv", dtype={"isbn": "string"})

    # Create current_quarter column
    df['current_quarter'] = 0
    df.loc[df['Quarter_num_4'] == 1, 'current_quarter'] = 4
    df.loc[df['Quarter_num_3'] == 1, 'current_quarter'] = 3
    df.loc[df['Quarter_num_2'] == 1, 'current_quarter'] = 2
    df.loc[df['Quarter_num_1'] == 1, 'current_quarter'] = 1

    # Define the three ISBNs
    isbn1 = "9781338896459"   # Dog Man #13
    isbn2 = "9781338896398"   # Cat Kid Comic Club #5
    isbn3 = "9781913484521"   # Third title

    filt1 = (df["isbn"] == isbn1) & (df["channel"] == 0)
    filt2 = (df["isbn"] == isbn2) & (df["channel"] == 0)
    filt3 = (df["isbn"] == isbn3) & (df["channel"] == 0)
    # Aggregate predicted sales (Next_Q1) by current quarter for each book
    sales1 = df[filt1].groupby("current_quarter")["Next_Q1"].sum()
    sales2 = df[filt2].groupby("current_quarter")["Next_Q1"].sum()
    sales3 = df[filt3].groupby("current_quarter")["Next_Q1"].sum()

    # Single figure
    plt.figure(figsize=(11, 6.5))

    # Blue line
    plt.plot(sales1.index, sales1.values,
             marker='o', linestyle='-', linewidth=2.8, color='tab:blue',
             label="Dog Man #13 (9781338896459)")

    # Orange line
    plt.plot(sales2.index, sales2.values,
             marker='s', linestyle='-', linewidth=2.8, color='tab:orange',
             label="Cat Kid Comic Club #5 (9781338896398)")

    # Green line
    plt.plot(sales3.index, sales3.values,
             marker='^', linestyle='-', linewidth=2.8, color='tab:green',
             label="Guiness Gamer’s 2025 (9781913484521)")

    # Formatting
    plt.title("Predicted Next-Quarter Sales (Next_Q1) for Three Titles",
              fontsize=15, pad=20)
    plt.xlabel("Current Quarter", fontsize=13)
    plt.ylabel("Predicted Sales Sum (Next_Q1)", fontsize=13)
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc="best")

    plt.tight_layout()
    plt.show()

    number_of_reviews1 = df[df["isbn"] == isbn1]["number_of_reviews"].iloc[0]
    number_of_reviews2 = df[df["isbn"] == isbn2]["number_of_reviews"].iloc[0]
    number_of_reviews3 = df[df["isbn"] == isbn3]["number_of_reviews"].iloc[0]

    titles = [
        "Dog Man #13\n(9781338896459)",
        "Cat Kid Comic Club #5\n(9781338896398)",
        "Guinness Gamer’s 2025\n(9781913484521)"
    ]
    reviews = [number_of_reviews1, number_of_reviews2, number_of_reviews3]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(titles, reviews, color=colors, edgecolor='black', linewidth=1.2)

    plt.title("Number of Reviews for Each Title", fontsize=15, pad=20)
    plt.ylabel("Number of Reviews", fontsize=13)
    plt.xticks(rotation=0, ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
    # ------------------------------------------------------------------

    print_length1 = df[df["isbn"] == isbn1]["print_length"].iloc[0]
    print_length2 = df[df["isbn"] == isbn2]["print_length"].iloc[0]
    print_length3 = df[df["isbn"] == isbn3]["print_length"].iloc[0]
    lengths = [print_length1, print_length2, print_length3]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(titles, lengths, color=colors, edgecolor='black',
                     linewidth=1.2)
    plt.title("Print Length for Each Title", fontsize=15, pad=20)
    plt.ylabel("Print Length (pages)", fontsize=13)
    plt.xticks(rotation=0, ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("end")