import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------#
#     Train_knight.csv         #
# -----------------------------#

def Train_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        sith = df[df['knight'] == 'Sith']
        jedi = df[df['knight'] == 'Jedi']
        def plot_scatter(ax, x, y, title):
            ax.scatter(jedi[x], jedi[y], color='blue', label='Jedi', alpha=0.7)
            ax.scatter(sith[x], sith[y], color='red', label='Sith', alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.6)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_scatter(axes[0], "Empowered", "Stims", "Separated (Empowered vs Stims)")
        plot_scatter(axes[1], "Deflection", "Push", "Mixed (Deflection vs Push)")
        plt.suptitle("Jedi vs Sith Attribute Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error {e}")




# -----------------------------#
#     Test_knight.csv         #
# -----------------------------#

def Test_knight(path_file):
    try:
        df_test = pd.read_csv(path_file)
        def plot_scatter(ax, x, y, title, color="green"):
            ax.scatter(
                df_test[x], df_test[y],
                color=color, label='knights', alpha=0.6, edgecolors="k"
            )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.6)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_scatter(axes[0], "Empowered", "Stims", "Test Set - Separated (Empowered vs Stims)")
        plot_scatter(axes[1], "Deflection", "Push", "Test Set - Mixed (Deflection vs Push)")
        plt.suptitle("Test Dataset Attribute Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
    except Exception as s:
        print(f"Error {e}")


def main():
    Train_knight("../Train_knight.csv")
    Test_knight("../Test_knight.csv")



if __name__ == "__main__":
    main()

        


