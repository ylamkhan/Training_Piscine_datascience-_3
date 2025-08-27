import pandas as pd

# -----------------------------#
#      Train_knight.csv        #
# -----------------------------#

def main(path_file):
    try:    
        df = pd.read_csv(path_file)
        df['knight'] = df['knight'].map({'Sith': 1, 'Jedi': 0})
        numeric_cols = df.select_dtypes(include='number').columns
        corr_series = df[numeric_cols].corrwith(df['knight'])
        corr_df = corr_series.reset_index()
        corr_df.columns = ['Feature', 'Correlation']
        corr_df['Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
        print(corr_df)
    except Exception as e:
        print(f"ERROR {e}")


if __name__ == "__main__":
    main("../Train_knight.csv")



