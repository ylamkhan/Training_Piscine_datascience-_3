import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



#-----------------------------------------------#
#                                               #
#                                               #
#         x(standardized) ​= ((​x−x') /σx)​        #
#                                               #
#  Where:                                       #
#       x  → original value                     #
#       x' → mean of the feature                #
#       σx → standard deviation of the feature  #
#-----------------------------------------------#


# ----------------------------------#
#         Test_knight.csv           #
# ----------------------------------#

def Test_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        features = df.select_dtypes(include="number").columns
        print("Original Data (first rows):")
        print(df[features].head(1), "\n")
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(
            scaler.fit_transform(df[features]),
            columns=features
        )
        print("Standardized Data (first rows):")
        print(df_standardized[features].head(1), "\n")
        plt.figure(figsize=(8,6))
        plt.scatter(df_standardized["Empowered"], df_standardized["Stims"], color="purple", alpha=0.7, label="knights")
        plt.xlabel("Empowered")
        plt.ylabel("Stims")
        plt.title("Standardized Data: Empowered vs Stims")
        plt.legend()
        plt.grid(False)
        plt.show()
    except Exception as e:
        print(f"Error {e}")



# ----------------------------------#
#         Train_knight.csv          #
# ----------------------------------#

def Train_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        features = df.select_dtypes(include="number").columns
        print("Original Data (first rows):")
        print(df[features].head(1), "\n")
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(
            scaler.fit_transform(df[features]),
            columns=features
        )
        for col in df.columns:
            if col not in features:
                df_standardized[col] = df[col]
        print("Standardized Data (first rows):")
        print(df_standardized[features].head(1), "\n")
        sith = df_standardized[df_standardized['knight'] == 'Sith']
        jedi = df_standardized[df_standardized['knight'] == 'Jedi']
        def plot_scatter(ax, x, y, title):
            ax.scatter(jedi[x], jedi[y], color='blue', label='Jedi', alpha=0.7)
            ax.scatter(sith[x], sith[y], color='red', label='Sith', alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title)
            ax.legend(loc="upper right") 
            ax.grid(False)
        fig, ax = plt.subplots(figsize=(8,6))
        plot_scatter(ax, "Empowered", "Stims", "Separated (Empowered vs Stims)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error {e}")



def main():
    Train_knight("../Train_knight.csv")
    Test_knight("../Test_knight.csv")

if __name__ == "__main__":
    main()



