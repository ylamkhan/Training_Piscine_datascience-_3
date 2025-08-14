import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KnightAnalyzer:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        
    def load_data(self, train_path='Train_knight.csv', test_path='Test_knight.csv'):
        """Load training and test datasets"""
        try:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            print(f"✅ Data loaded successfully!")
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    # Exercise 00: Histogram
    def create_histograms(self, save_plots=True):
        """Create histograms for all features comparing Jedi vs Sith"""
        if self.train_data is None:
            print("❌ Please load data first!")
            return
        
        # Get feature columns (exclude target)
        features = [col for col in self.train_data.columns if col != 'knight']
        
        # Create subplots
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Plot histograms for each class
            jedi_data = self.train_data[self.train_data['knight'] == 'Jedi'][feature]
            sith_data = self.train_data[self.train_data['knight'] == 'Sith'][feature]
            
            ax.hist(jedi_data, alpha=0.7, label='Jedi', bins=15, color='lightblue')
            ax.hist(sith_data, alpha=0.7, label='Sith', bins=15, color='red')
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('histograms_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Exercise 00 completed: Histograms created!")
    
    # Exercise 01: Correlation
    def analyze_correlation(self):
        """Calculate and display correlation between features and target"""
        if self.train_data is None:
            print("❌ Please load data first!")
            return
        
        # Convert target to numeric for correlation
        train_encoded = self.train_data.copy()
        train_encoded['knight_numeric'] = train_encoded['knight'].map({'Jedi': 1, 'Sith': 0})
        
        # Calculate correlations with target
        correlations = train_encoded.drop(['knight'], axis=1).corr()['knight_numeric'].sort_values(ascending=False)
        
        print("🔍 Correlation Analysis - Features vs Knight Type:")
        print("="*50)
        for feature, corr in correlations.items():
            if feature != 'knight_numeric':
                print(f"{feature:<15}: {corr:.6f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = train_encoded.drop(['knight'], axis=1).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Exercise 01 completed: Correlation analysis done!")
        return correlations
    
    # Exercise 02: Scatter plots
    def create_scatter_plots(self, feature1='Empowered', feature2='Stims'):
        """Create scatter plots - one separating clusters, one mixing them"""
        if self.train_data is None:
            print("❌ Please load data first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training data - separated clusters
        jedi_train = self.train_data[self.train_data['knight'] == 'Jedi']
        sith_train = self.train_data[self.train_data['knight'] == 'Sith']
        
        axes[0,0].scatter(jedi_train[feature1], jedi_train[feature2], 
                         c='lightblue', alpha=0.7, label='Jedi', s=50)
        axes[0,0].scatter(sith_train[feature1], sith_train[feature2], 
                         c='red', alpha=0.7, label='Sith', s=50)
        axes[0,0].set_title(f'Training Data - {feature1} vs {feature2} (Separated)')
        axes[0,0].set_xlabel(feature1)
        axes[0,0].set_ylabel(feature2)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Training data - mixed clusters (using different features)
        feature3, feature4 = 'Mass', 'Survival'  # These have low correlation
        axes[0,1].scatter(self.train_data[feature3], self.train_data[feature4], 
                         c=np.where(self.train_data['knight'] == 'Jedi', 'lightblue', 'red'),
                         alpha=0.7, s=50)
        axes[0,1].set_title(f'Training Data - {feature3} vs {feature4} (Mixed)')
        axes[0,1].set_xlabel(feature3)
        axes[0,1].set_ylabel(feature4)
        axes[0,1].grid(True, alpha=0.3)
        
        # Test data plots (no target available, so single color)
        axes[1,0].scatter(self.test_data[feature1], self.test_data[feature2], 
                         c='green', alpha=0.7, s=50, label='Test Data')
        axes[1,0].set_title(f'Test Data - {feature1} vs {feature2}')
        axes[1,0].set_xlabel(feature1)
        axes[1,0].set_ylabel(feature2)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].scatter(self.test_data[feature3], self.test_data[feature4], 
                         c='purple', alpha=0.7, s=50, label='Test Data')
        axes[1,1].set_title(f'Test Data - {feature3} vs {feature4}')
        axes[1,1].set_xlabel(feature3)
        axes[1,1].set_ylabel(feature4)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Exercise 02 completed: Scatter plots created!")
    
    # Exercise 03: Standardization
    def standardize_data(self, display_plot=True):
        """Standardize the data (mean=0, std=1)"""
        if self.train_data is None:
            print("❌ Please load data first!")
            return None, None
        
        # Get feature columns
        feature_cols = [col for col in self.train_data.columns if col != 'knight']
        
        # Original data
        print("Original Training Data (first 3 rows):")
        print(self.train_data[feature_cols].head(3))
        print()
        
        # Fit and transform training data
        train_features_scaled = self.scaler_standard.fit_transform(self.train_data[feature_cols])
        train_standardized = pd.DataFrame(train_features_scaled, columns=feature_cols)
        train_standardized['knight'] = self.train_data['knight'].values
        
        # Transform test data
        test_features_scaled = self.scaler_standard.transform(self.test_data[feature_cols])
        test_standardized = pd.DataFrame(test_features_scaled, columns=feature_cols)
        
        print("Standardized Training Data (first 3 rows):")
        print(train_standardized[feature_cols].head(3))
        print()
        
        if display_plot:
            # Create scatter plot with standardized data
            plt.figure(figsize=(10, 6))
            jedi_data = train_standardized[train_standardized['knight'] == 'Jedi']
            sith_data = train_standardized[train_standardized['knight'] == 'Sith']
            
            plt.scatter(jedi_data['Empowered'], jedi_data['Stims'], 
                       c='lightblue', alpha=0.7, label='Jedi', s=50)
            plt.scatter(sith_data['Empowered'], sith_data['Stims'], 
                       c='red', alpha=0.7, label='Sith', s=50)
            
            plt.title('Standardized Data - Empowered vs Stims')
            plt.xlabel('Empowered (standardized)')
            plt.ylabel('Stims (standardized)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('standardized_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Exercise 03 completed: Data standardized!")
        return train_standardized, test_standardized
    
    # Exercise 04: Normalization
    def normalize_data(self, display_plot=True):
        """Normalize the data (min-max scaling to 0-1 range)"""
        if self.train_data is None:
            print("❌ Please load data first!")
            return None, None
        
        # Get feature columns
        feature_cols = [col for col in self.train_data.columns if col != 'knight']
        
        # Original data
        print("Original Training Data (first 3 rows):")
        print(self.train_data[feature_cols].head(3))
        print()
        
        # Fit and transform training data
        train_features_normalized = self.scaler_minmax.fit_transform(self.train_data[feature_cols])
        train_normalized = pd.DataFrame(train_features_normalized, columns=feature_cols)
        train_normalized['knight'] = self.train_data['knight'].values
        
        # Transform test data
        test_features_normalized = self.scaler_minmax.transform(self.test_data[feature_cols])
        test_normalized = pd.DataFrame(test_features_normalized, columns=feature_cols)
        
        print("Normalized Training Data (first 3 rows):")
        print(train_normalized[feature_cols].head(3))
        print()
        
        if display_plot:
            # Create scatter plot with normalized data
            plt.figure(figsize=(10, 6))
            jedi_data = train_normalized[train_normalized['knight'] == 'Jedi']
            sith_data = train_normalized[train_normalized['knight'] == 'Sith']
            
            plt.scatter(jedi_data['Mass'], jedi_data['Survival'], 
                       c='lightblue', alpha=0.7, label='Jedi', s=50)
            plt.scatter(sith_data['Mass'], sith_data['Survival'], 
                       c='red', alpha=0.7, label='Sith', s=50)
            
            plt.title('Normalized Data - Mass vs Survival')
            plt.xlabel('Mass (normalized)')
            plt.ylabel('Survival (normalized)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('normalized_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Exercise 04 completed: Data normalized!")
        return train_normalized, test_normalized
    
    # Exercise 05: Split data
    def split_data(self, test_size=0.2, random_state=42):
        """Split training data into training and validation sets"""
        if self.train_data is None:
            print("❌ Please load data first!")
            return None, None
        
        # Split the data
        train_part, val_part = train_test_split(self.train_data, 
                                               test_size=test_size, 
                                               random_state=random_state, 
                                               stratify=self.train_data['knight'])
        
        # Save to files
        train_part.to_csv('Training_knight.csv', index=False)
        val_part.to_csv('Validation_knight.csv', index=False)
        
        print(f"📊 Data Split Summary:")
        print(f"Original training data: {len(self.train_data)} samples")
        print(f"New training set: {len(train_part)} samples ({(1-test_size)*100:.0f}%)")
        print(f"Validation set: {len(val_part)} samples ({test_size*100:.0f}%)")
        print()
        print("Class distribution in splits:")
        print("Training set:")
        print(train_part['knight'].value_counts(normalize=True))
        print("\nValidation set:")
        print(val_part['knight'].value_counts(normalize=True))
        
        print("\n✅ Exercise 05 completed: Data split and saved!")
        print("💾 Files saved: Training_knight.csv, Validation_knight.csv")
        
        return train_part, val_part
    
    def comprehensive_analysis(self):
        """Run all analyses"""
        print("🚀 Starting Comprehensive Knight Analysis")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return
        
        print("\n" + "="*60)
        
        # Exercise 00: Histograms
        print("📊 Exercise 00: Creating Histograms...")
        self.create_histograms()
        
        print("\n" + "="*60)
        
        # Exercise 01: Correlation
        print("🔍 Exercise 01: Analyzing Correlations...")
        correlations = self.analyze_correlation()
        
        print("\n" + "="*60)
        
        # Exercise 02: Scatter plots
        print("📈 Exercise 02: Creating Scatter Plots...")
        self.create_scatter_plots()
        
        print("\n" + "="*60)
        
        # Exercise 03: Standardization
        print("⚖️ Exercise 03: Standardizing Data...")
        train_std, test_std = self.standardize_data()
        
        print("\n" + "="*60)
        
        # Exercise 04: Normalization
        print("📏 Exercise 04: Normalizing Data...")
        train_norm, test_norm = self.normalize_data()
        
        print("\n" + "="*60)
        
        # Exercise 05: Split
        print("✂️ Exercise 05: Splitting Data...")
        train_split, val_split = self.split_data()
        
        print("\n" + "="*60)
        print("🎉 All exercises completed successfully!")
        print("📁 Generated files:")
        print("   • histograms_comparison.png")
        print("   • correlation_heatmap.png")
        print("   • scatter_plots.png")
        print("   • standardized_plot.png")
        print("   • normalized_plot.png")
        print("   • Training_knight.csv")
        print("   • Validation_knight.csv")

# Usage example
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = KnightAnalyzer()
    
    # Run comprehensive analysis
    analyzer.comprehensive_analysis()
    
    # Additional insights for large datasets
    print("\n🔮 Additional Insights for Large Datasets:")
    print("="*60)
    print("1. Feature Engineering: Consider creating interaction features")
    print("2. Dimensionality Reduction: Use PCA for high-dimensional data")
    print("3. Model Selection: Try ensemble methods for better performance")
    print("4. Cross-validation: Use k-fold CV for robust evaluation")
    print("5. Feature Selection: Remove low-correlation features")
    print("6. Data Quality: Check for outliers and missing values")
    print("7. Scaling: Choose between standardization and normalization based on distribution")
    print("8. Validation Strategy: Ensure stratified splits for balanced classes")