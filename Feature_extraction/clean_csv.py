from math import comb
import pandas as pd
import argparse
import json

def clean_missing_values(train_csv, val_csv, test_csv, output_train_csv, output_val_csv, output_test_csv, columns_to_fix):
    """
    Replaces 'None' values in specified columns with the column's max value across all three datasets.
    If a column has only 'None', replaces them with -1.

    :param train_csv: Path to input training CSV file
    :param val_csv: Path to input validation CSV file
    :param test_csv: Path to input test CSV file
    :param output_train_csv: Path to save the cleaned training CSV file
    :param output_val_csv: Path to save the cleaned validation CSV file
    :param output_test_csv: Path to save the cleaned test CSV file
    :param columns_to_fix: List of column names to process
    """

    df_train = pd.read_csv(train_csv)
    df_train["decade"] = df_train["decade"].astype(int)
    df_train.to_csv(train_csv, index=False)
    
    df_val = pd.read_csv(val_csv)
    df_val["decade"] = df_val["decade"].astype(int)
    df_val.to_csv(val_csv, index=False)
    
    df_test = pd.read_csv(test_csv)
    df_test["decade"] = df_test["decade"].astype(int)
    df_test.to_csv(test_csv, index=False)

    combined_df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    changed_columns = 0
    missing_columns = []
    
    for col in columns_to_fix:
        if col in combined_df.columns:
            original_nulls = combined_df[col].isin(["None"]).sum()

            combined_df[col] = combined_df[col].replace("None", pd.NA).astype("float64")

            max_val = combined_df[col].max(skipna=True)

            if pd.isna(max_val):
                print(f"Column '{col}' has only None values. Replacing with -1.")
                max_val = -1
            else:
                print(f"Replacing None in column '{col}' with max value: {max_val}")

            if original_nulls > 0:
                changed_columns += 1

            df_train[col] = df_train[col].replace("None", pd.NA).fillna(max_val)
            df_val[col] = df_val[col].replace("None", pd.NA).fillna(max_val)
            df_test[col] = df_test[col].replace("None", pd.NA).fillna(max_val)
        else:
            missing_columns.append(col)

    print(f"Total columns changed: {changed_columns}")
    print(f"Columns not found in the dataset: {len(missing_columns)}")
    if missing_columns:
        print("Missing columns:")
        for col in missing_columns:
            print(f" - {col}")

    df_train.to_csv(output_train_csv, index=False)
    df_val.to_csv(output_val_csv, index=False)
    df_test.to_csv(output_test_csv, index=False)

    print(f"Cleaned datasets saved to {output_train_csv}, {output_val_csv}, and {output_test_csv}")
    
def clean_missing_values_gutenberg(gutenberg_csv: str, output_gutenberg_csv: str, text_info_path: str):
    """
    Cleans a Gutenberg CSV by replacing 'None' values in target word columns with the column's max value.
    If a column has only 'None', replaces them with -1. Updates the file in-place.

    Args:
        csv_path (str): Path to the Gutenberg CSV.
        text_info_path (str): Path to the text_info CSV with 'file_name' and 'decade'.
    """
    df = pd.read_csv(gutenberg_csv)

    if "file_name" not in df.columns:
        print("The CSV must contain a 'file_name' column.", flush=True)
        return

    if "decade" not in df.columns:
        df["decade"] = None

    df["decade"] = pd.to_numeric(df["decade"], errors="coerce")
    missing_mask = df["decade"].isna() | ~df["decade"].apply(lambda x: float(x).is_integer())

    if not missing_mask.any():
        print("All rows have valid 'decade' values.", flush=True)
        return

    print(f"Found {missing_mask.sum()} rows with missing or invalid 'decade'. Attempting to fix...", flush=True)

    text_info = pd.read_csv(text_info_path)[["file_name", "decade"]]
    text_dict = dict(zip(text_info["file_name"], text_info["decade"]))

    df.loc[missing_mask, "decade"] = df.loc[missing_mask, "file_name"].map(text_dict)

    still_missing = df[df["decade"].isna()]
    if not still_missing.empty:
        print(f"Could not fix {len(still_missing)} rows. They are still missing 'decade':", flush=True)
        print(still_missing[["file_name"]])
    else:
        print("Successfully fixed all missing 'decade' values.", flush=True)
    
    df.to_csv(output_gutenberg_csv, index=False)
    print(f"Updated CSV saved to: {output_gutenberg_csv}", flush=True)


def check_and_fix_missing_decades(csv_path: str, text_info_path: str):
    """
    Checks for missing or invalid 'decade' values in a feature CSV and fills them
    using a reference text_info CSV. Updates the file in-place.

    Args:
        csv_path (str): Path to the feature CSV.
        text_info_path (str): Path to the text_info CSV with 'file_name' and 'decade'.
    """
    df = pd.read_csv(csv_path)


    if "file_name" not in df.columns:
        print("The CSV must contain a 'file_name' column.", flush=True)
        return
    

    if "decade" not in df.columns:
        df["decade"] = None


    df["decade"] = pd.to_numeric(df["decade"], errors="coerce")
    missing_mask = df["decade"].isna() | ~df["decade"].apply(lambda x: float(x).is_integer())

    if not missing_mask.any():
        print("All rows have valid 'decade' values.", flush=True)
        return

    print(f"Found {missing_mask.sum()} rows with missing or invalid 'decade'. Attempting to fix...", flush=True)

    text_info = pd.read_csv(text_info_path)[["file_name", "decade"]]
    text_dict = dict(zip(text_info["file_name"], text_info["decade"]))

    df.loc[missing_mask, "decade"] = df.loc[missing_mask, "file_name"].map(text_dict)

    still_missing = df[df["decade"].isna()]
    if not still_missing.empty:
        print(f"Could not fix {len(still_missing)} rows. They are still missing 'decade':", flush=True)
        print(still_missing[["file_name"]])
    else:
        print("Successfully fixed all missing 'decade' values.", flush=True)

    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to: {csv_path}", flush=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace None values in target word features with max values")
    parser.add_argument("--text_info", type=str, required=True, help="Path to the text information CSV file")
    
    parser.add_argument("--input_train_csv", type=str, required=False, help="Path to the input training CSV file")
    parser.add_argument("--input_validation_csv", type=str, required=False, help="Path to the input validation CSV file")
    parser.add_argument("--input_test_csv", type=str, required=False, help="Path to the input test CSV file")
    parser.add_argument("--input_gutenberg_csv", type=str, required=False, help="Path to the input Gutenberg CSV file")

    parser.add_argument("--output_train_csv", type=str, required=False, help="Path to save the cleaned training CSV file")
    parser.add_argument("--output_validation_csv", type=str, required=False, help="Path to save the cleaned validation CSV file")
    parser.add_argument("--output_test_csv", type=str, required=False, help="Path to save the cleaned test CSV file")
    parser.add_argument("--output_gutenberg_csv", type=str, required=False, help="Path to save the cleaned Gutenberg CSV file")

    parser.add_argument("--lang", type=str, required=True, help="Language used for target word extraction")
    parser.add_argument("--target_words_json", type=str, required=True, help="Path to JSON containing target word lists")

    args = parser.parse_args()
       
    if args.input_train_csv:
        check_and_fix_missing_decades(args.input_train_csv, args.text_info)
    if args.input_validation_csv:    
        check_and_fix_missing_decades(args.input_validation_csv, args.text_info)
    if args.input_test_csv:
        check_and_fix_missing_decades(args.input_test_csv, args.text_info)
    if args.input_gutenberg_csv:
        check_and_fix_missing_decades(args.input_gutenberg_csv, args.text_info)


    with open(args.target_words_json, "r", encoding="utf-8") as f:
        word_config = json.load(f)

    if args.lang not in word_config:
        raise ValueError(f"Language '{args.lang}' not found in {args.target_words_json}")

    target_words = word_config[args.lang]
    target_columns = list(target_words)

    print(f"Cleaning target word columns for language '{args.lang}':", flush=True)
    for word in target_columns:
        print(f" - {word}", flush=True)
        
    if args.input_train_csv and args.input_validation_csv and args.input_test_csv and args.output_train_csv and args.output_validation_csv and args.output_test_csv:
        clean_missing_values(
            args.input_train_csv, args.input_validation_csv, args.input_test_csv,
            args.output_train_csv, args.output_validation_csv, args.output_test_csv,
            target_columns
        )
        
    if args.input_gutenberg_csv and args.output_gutenberg_csv:
        clean_missing_values_gutenberg(
            args.input_gutenberg_csv, args.output_gutenberg_csv, args.text_info
        )
    
