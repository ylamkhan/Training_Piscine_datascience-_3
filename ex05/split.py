#!/usr/bin/env python3
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------#
#      Train_knight.csv        #
# -----------------------------#


def split():
    try:
        if len(sys.argv) != 2:
            print("Usage: ./split.py Train_knight.csv")
            sys.exit(1)
        input_file = sys.argv[1]
        if input_file != "../Train_knight.csv":
            print("Name the file incorrect")
            sys.exit(0)
        df = pd.read_csv(input_file)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        train_df.to_csv("Training_knight.csv", index=False)
        val_df.to_csv("Validation_knight.csv", index=False)
        # print(f"Original dataset size: {len(df)}")
        # print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        # print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"Error {e}")


if __name__ == "__main__":
    split()
    

