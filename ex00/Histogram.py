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
            ax = axes[i]
            ax.hist(sith[col], bins=40, alpha=0.6, color='#ff5252', label='Sith')
            ax.hist(jedi[col], bins=40, alpha=0.6, color='#0000FF', label='Jedi')
            ax.set_title(col)
            ax.grid(False)
            ax.legend(loc='upper right')
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
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
        category_col = df.columns[-1]
        axes = df.select_dtypes(include="number").hist(
            figsize=(15, 12),
            color='#83f28f',
            bins = 60
        )
        for ax in np.ravel(axes):
            ax.grid(False)
        for ax in np.ravel(axes):
            ax.legend(['knight'], loc="upper right")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"ERROR {e}")


def main():
    Train_knight("../Train_knight.csv")
    Test_knight("../Test_knight.csv")



if __name__ == "__main__":
    main()
