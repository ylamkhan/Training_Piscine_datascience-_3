#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#--------------------------------------------------#
#           Xnarm = (X - Xmin) / (Xmax - Xmin)     #
#--------------------------------------------------#


#----------------------------------#
#          Train_knight.csv        #
#----------------------------------#


def Train_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        fea = df.select_dtypes(include="number").columns
        print("Before normalization:\n", df.head(1))
        features = df.drop(columns=['knight'])
        scaler = MinMaxScaler()
        features_norm = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        print("\nAfter normalization:\n", features_norm.head(1))
        for col in df.columns:
            if col not in fea:
                features_norm[col] = df[col]
        sith = features_norm[features_norm['knight'] == 'Sith']
        jedi = features_norm[features_norm['knight'] == 'Jedi']
        plt.figure()
        plt.scatter(sith['Deflection'], sith['Push'], color='red', label='Sith', alpha=0.6)
        plt.scatter(jedi['Deflection'], jedi['Push'], color='blue', label='Jedi', alpha=0.6)
        plt.xlabel('Deflection')
        plt.ylabel('Push')
        plt.title('Train_knight - Deflection vs Push')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error {e}")


#----------------------------------#
#          Tesr_knight.csv        #
#----------------------------------#

    
def Test_knight(path_file):
    try:
        df = pd.read_csv(path_file)
        print("Before normalization:\n", df.head())
        scaler = MinMaxScaler()
        features_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        print("\nAfter normalization:\n", features_norm.head())
        plt.figure()
        plt.scatter(features_norm['Deflection'], features_norm['Push'], color='green', label='Knights', alpha=0.6)
        plt.xlabel('Deflection')
        plt.ylabel('Push')
        plt.title('Tesr_knight - Deflection vs Push')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error {e}")



def main():
    Train_knight("../Train_knight.csv")
    Test_knight("../Test_knight.csv")


if __name__ == "__main__":
    main()
