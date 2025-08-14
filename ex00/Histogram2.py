import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        df = pd.read_csv('../Train_knight.csv')
        print(df.head())

        # numeric_cols = df.select_dtypes(include="number").columns

        # sith = df[df['knight'] == 'Sith']
        # jedi = df[df['knight'] == 'Jedi']

        # # Create subplots
        # fig, axes = plt.subplots(nrows=int(np.ceil(len(numeric_cols)/5)), ncols=5, figsize=(15, 15))
        # axes = axes.flatten()

        # for i, col in enumerate(numeric_cols):
        #     ax = axes[i]
        #     ax.hist(sith[col], bins=40, alpha=0.6, color='#ff5252', label='Sith')
        #     ax.hist(jedi[col], bins=40, alpha=0.6, color='#0000FF', label='Jedi')
        #     ax.set_title(col)
        #     ax.grid(False)
        #     ax.legend(loc='upper right')

        # # Hide extra axes if any
        # for j in range(i+1, len(axes)):
        #     axes[j].axis('off')

        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"ERROR {e}")

if __name__ == "__main__":
    main()
