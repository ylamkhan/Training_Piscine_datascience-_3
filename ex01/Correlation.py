import pandas as pd
df = pd.read_csv('../Train_knight.csv')
df['knight_binary'] = df['knight'].map({'Sith': 1, 'Jedi': 0})
numeric_cols = df.select_dtypes(include='number').columns
corr_series = df[numeric_cols].corrwith(df['knight_binary'])
corr_df = corr_series.reset_index()
corr_df.columns = ['Feature', 'Correlation']
corr_df['Correlation'] = corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
print(corr_df)
