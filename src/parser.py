# Parse and clean .dat files

"""
parser.py

This script handles the parsing and cleaning of raw .DAT property sales files 
from the NSW Bulk Property Sales Information dataset.

Steps performed:
1. Recursively searches all year/week folders in the specified base directory.
2. Opens each `.DAT` file and extracts rows that start with 'B;', 
   which contain core sale transaction data.
3. Parses fields like address, suburb, sale price, contract date, etc.
4. Cleans the data: drops missing values, standardizes suburb names, 
   converts dates and prices to numeric format.
5. Filters only the top 20 most popular suburbs in NSW (configurable).
6. Returns a clean pandas DataFrame for downstream processing.
"""

import os
import pandas as pd
from glob import glob
from tqdm import tqdm

TOP_SUBURBS = [
    'Sydney', 'Parramatta', 'Newtown', 'Bondi', 'Chatswood',
    'Liverpool', 'Blacktown', 'Campbelltown', 'Penrith', 'Manly',
    'Strathfield', 'Hornsby', 'Castle Hill', 'Burwood', 'Wollongong',
    'Wetherill Park', 'Baulkham Hills', 'Bankstown', 'Ryde', 'Auburn'
]

def extract_b_records_from_file(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("B;"):
                parts = line.strip().split(';')
                try:
                    record = {
                        "Street_No": parts[7],
                        "Street_Name": parts[8],
                        "Suburb": parts[9].title().strip(),
                        "Postcode": parts[10],
                        "Land_Size_sqm": float(parts[11]) if parts[11] else None,
                        "Contract_Date": parts[13],
                        "Settlement_Date": parts[14],
                        "Sale_Price": float(parts[15]) if parts[15] else None,
                        "Zone": parts[16],
                        "Property_Use": parts[17],
                        "Description": parts[18]
                    }
                    records.append(record)
                except Exception as e:
                    print(f"Skipped line due to error: {e}")
    return records

def parse_all_files(base_dir="data/raw"):
    all_records = []

    for year in sorted(os.listdir(base_dir)):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        week_folders = sorted(os.listdir(year_path))
        for week in week_folders:
            week_path = os.path.join(year_path, week)
            dat_files = glob(os.path.join(week_path, "*.DAT"))

            for file in dat_files:
                all_records.extend(extract_b_records_from_file(file))

    return pd.DataFrame(all_records)

def clean_and_filter(df):
    df['Sale_Price'] = pd.to_numeric(df['Sale_Price'], errors='coerce')
    df['Contract_Date'] = pd.to_datetime(df['Contract_Date'], format='%Y%m%d', errors='coerce')
    df['Settlement_Date'] = pd.to_datetime(df['Settlement_Date'], format='%Y%m%d', errors='coerce')

    df = df.dropna(subset=['Suburb', 'Sale_Price'])
    df = df[df['Suburb'].isin([s.title() for s in TOP_SUBURBS])]

    # Create full address
    df['Address'] = df['Street_No'].astype(str) + " " + df['Street_Name']
    return df

def save_clean_csv(df, output_path="data/clean_sales_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}\n")


