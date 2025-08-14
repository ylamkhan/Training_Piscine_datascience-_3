import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        df = pd.read_csv('../Test_knight.csv')
        # print(df.head())
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

if __name__ == "__main__":
    main()
