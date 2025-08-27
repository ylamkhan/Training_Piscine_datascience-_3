import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------------------------#
#         Test_knight.csv           #
# ----------------------------------#

def Test_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        features = df.select_dtypes(include=['float64', 'int64']).columns
        print("Original Data (first rows):")
        print(df[features].head(), "\n")
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(
            scaler.fit_transform(df[features]),
            columns=features
        )
        for col in df.columns:
            if col not in features:
                df_standardized[col] = df[col]
        print("Standardized Data (first rows):")
        print(df_standardized[features].head(), "\n")
        plt.figure(figsize=(8,6))
        plt.scatter(df_standardized["Empowered"], df_standardized["Stims"], 
                    color="purple", alpha=0.7, label="knights")
        plt.xlabel("Empowered (Standardized)")
        plt.ylabel("Stims (Standardized)")
        plt.title("Standardized Data: Empowered vs Stims")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    except Exception as e:
        print(f"Error {e}")



# ----------------------------------#
#         Train_knight.csv           #
# ----------------------------------#

def Train_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        features = df.select_dtypes(include=['float64', 'int64']).columns
        print("Original Data (first rows):")
        print(df[features].head(), "\n")
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(
            scaler.fit_transform(df[features]),
            columns=features
        )
        for col in df.columns:
            if col not in features:
                df_standardized[col] = df[col]
        print("Standardized Data (first rows):")
        print(df_standardized[features].head(), "\n")
        plt.figure(figsize=(8,6))
        for knight in df_standardized["knight"].unique():
            subset = df_standardized[df_standardized["knight"] == knight]
            plt.scatter(subset["Empowered"], subset["Stims"], label=knight, alpha=0.7)
        plt.xlabel("Empowered (Standardized)")
        plt.ylabel("Stims (Standardized)")
        plt.title("Standardized Data: Empowered vs Stims")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    except Exception as e:
        print(f"Error {e}")



def main():
    Train_knight("../Train_knight.csv")
    Test_knight("../Test_knight.csv")

if __name__ == "__main__":
    main()



