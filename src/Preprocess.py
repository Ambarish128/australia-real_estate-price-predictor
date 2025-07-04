"""
Main script to process raw real estate data files and produce a cleaned CSV.

Steps:
1. Parse all raw .DAT files from the 'data/raw' directory into a single DataFrame.
2. Clean and filter the data to focus on top suburbs.
3. (Optional) Calculate distance to Sydney CBD (currently commented out).
4. Save the cleaned and filtered data as a CSV file for further processing.

Author: Ambarish Shashank Gadgil
Date: 2025-07-04
"""

from parser import parse_all_files, clean_and_filter, save_clean_csv



def main():
    print("📥 Parsing raw .DAT files...")
    raw_df = parse_all_files("data/raw")

    print("🧹 Cleaning and filtering for top suburbs...")
    clean_df = clean_and_filter(raw_df)

    # print("📍 Calculating distance to Sydney CBD...")
    # clean_df = add_distance_to_cbd(clean_df)

    #Saving the final CSV file
    save_clean_csv(clean_df)

    

if __name__ == "__main__":
    main()
