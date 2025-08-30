import pandas as pd

# -----------------------------#
#      Train_knight.csv        #
# -----------------------------#

#---------------------------------------------#
#                                             #
#                    cov(X,Y)​                 #
#               r = ----------                #
#                    σx . σy                  #
# Where:                                      #
#   . cov(X,Y) = covariance between X and Y​   #
#   . σx = standard deviation of X            #
#   . σy = standard deviation of Y            #
#---------------------------------------------#


#---------------------------------------------#
#                                             #
#                                             #
#     cov(X,Y) = 1/n .( ∑ ​(xi​− 'x)(yi​− 'y​))   #
#                                             #
#     σx  = 1/n . (∑ ​(xi​− 'x)^1/2)            #
#                                             #
# where:                                      #
#      'x = 1 / n . (∑ ​xi)​                    #
#---------------------------------------------#




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



