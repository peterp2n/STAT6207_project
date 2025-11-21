import sqlite3
import pandas as pd
import os

print("="*80)
print("STEP 1: Converting SQLite Database to CSV Files")
print("="*80 + "\n")

# Database file path (in parent directory)
db_path = "../Topic1_dataset.sqlite"

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# Get all table names (excluding internal sqlite tables)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
tables = cursor.fetchall()

print(f"Found {len(tables)} table(s) in database\n")

if len(tables) == 0:
    print(" No tables found in database!")
    print("Database might be empty or corrupted.")
    conn.close()
    exit(1)

# Convert each table to CSV
for table in tables:
    table_name = table[0]
    print(f"Processing table: {table_name}")
    
    # Read table into DataFrame
    df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
    
    # Output CSV filename (in same directory)
    csv_filename = f"{table_name}.csv"
    
    # Write to CSV
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"   Created: {csv_filename}")
    print(f"     Rows: {len(df)}")
    print(f"     Columns: {len(df.columns)}\n")

# Close connection
conn.close()

print("="*80)
print(f" Conversion complete! Created {len(tables)} CSV file(s)")
print("="*80)
