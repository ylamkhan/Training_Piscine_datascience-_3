import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------#
#      Train_knight.csv        #
# -----------------------------#


def Train_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        numeric_cols = df.select_dtypes(include="number").columns
        sith = df[df['knight'] == 'Sith']
        jedi = df[df['knight'] == 'Jedi']
        fig, axes = plt.subplots(nrows=int(np.ceil(len(numeric_cols)/5)), ncols=5, figsize=(15, 15))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(sith[col], bins=60, alpha=0.6, color='#ff5252', label='Sith')
            axes[i].hist(jedi[col], bins=60, alpha=0.6, color='#0000FF', label='Jedi')
            axes[i].set_title(col)
            axes[i].grid(False)
            axes[i].legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"ERROR {e}")

# -----------------------------#
#      Test_knight.csv         #
# -----------------------------#

def Test_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        numeric_cols = df.select_dtypes(include="number").columns
        ncols = 5
        nrows = int(np.ceil(len(numeric_cols)/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col], bins=60, alpha=0.6, color='green', label='knight')
            axes[i].set_title(col)
            axes[i].grid(False)
            axes[i].legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"ERROR {e}")



def main():
    Train_knight("../Train_knight.csv")
    Test_knight("../Test_knight.csv")



if __name__ == "__main__":
    main()
