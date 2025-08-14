import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv('../Train_knight.csv')
test_df = pd.read_csv('../Test_knight.csv')

# Choose two features for plotting (you can change them)
x_feature = 'Empowered'
y_feature = 'Stims'

# Define colors for knights
colors = {'Sith': 'red', 'Jedi': 'blue'}

# --------- Train graphs ---------
plt.figure(figsize=(12, 5))

# Graph 1: Train with clusters (colored by knight)
plt.subplot(1, 2, 1)
for knight in train_df['knight'].unique():
    subset = train_df[train_df['knight'] == knight]
    plt.scatter(subset[x_feature], subset[y_feature], label=knight, color=colors[knight], alpha=0.7)
plt.title('Train - Knights separated')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.legend()

# Graph 2: Train mixed (all points same color)
plt.subplot(1, 2, 2)
plt.scatter(train_df[x_feature], train_df[y_feature], color='grey', alpha=0.7)
plt.title('Train - Knights mixed')
plt.xlabel(x_feature)
plt.ylabel(y_feature)

plt.tight_layout()
plt.show()

# --------- Test graphs ---------
plt.figure(figsize=(12, 5))

# Graph 3: Test - colored by knight if exists, otherwise all same color
plt.subplot(1, 2, 1)
if 'knight' in test_df.columns:
    colors = {'Sith': 'red', 'Jedi': 'blue'}
    for knight in test_df['knight'].unique():
        subset = test_df[test_df['knight'] == knight]
        plt.scatter(subset[x_feature], subset[y_feature], label=knight, color=colors[knight], alpha=0.7)
    plt.legend()
else:
    plt.scatter(test_df[x_feature], test_df[y_feature], color='grey', alpha=0.7)
plt.title('Test - Knights separated (or single cluster)')
plt.xlabel(x_feature)
plt.ylabel(y_feature)

# Graph 4: Test mixed
plt.subplot(1, 2, 2)
plt.scatter(test_df[x_feature], test_df[y_feature], color='grey', alpha=0.7)
plt.title('Test - Knights mixed')
plt.xlabel(x_feature)
plt.ylabel(y_feature)

plt.tight_layout()
plt.show()
