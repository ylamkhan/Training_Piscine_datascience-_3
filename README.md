# Training Piscine Data Science - Module 3: "The Present"

## Project Overview

This module focuses on data analysis and visualization using a Star Wars themed dataset. You'll analyze Jedi/Sith knight data to understand patterns and predict which side of the Force a knight belongs to based on their skills and attributes.

### Theme
You're analyzing whether we could have predicted Anakin Skywalker's fall to the dark side by examining historical data of all knights and their characteristics.

## Dataset Description

The project uses two main CSV files:
- **Train_knight.csv**: Training dataset with knight features and their Force alignment
- **Test_knight.csv**: Test dataset for validation

### Features (Knight Skills)
The dataset contains approximately 30 different attributes representing knight abilities:

**High Correlation Features** (most predictive of Force alignment):
- Empowered (0.79)
- Stims (0.78)
- Prescience (0.78)
- Recovery (0.78)
- Strength (0.74)
- Sprint (0.73)
- Sensitivity (0.73)
- Power (0.71)

**Medium Correlation Features**:
- Awareness, Attunement, Dexterity, Delay, Slash, Force, Lightsaber, Evade, Combo, Burst, Hability, Blocking, Agility, Reactivity, Grasping, Repulse, Friendship

**Low Correlation Features**:
- Mass, Survival, Midi-chlorien, Push, Deflection

**Target Variable**: `knight` column (indicates Light/Dark side affiliation)

## Exercise Breakdown

### Exercise 00: Histogram Analysis
**Objective**: Create visualizations to understand data distribution

**Requirements**:
- Generate histograms for both Train_knight.csv and Test_knight.csv
- Show distribution of features
- Compare feature distributions between Light and Dark side knights
- Create overlaid histograms showing both classes

**Key Insights to Discover**:
- Which features show clear separation between Light/Dark knights
- Overall data distribution patterns
- Potential outliers or anomalies

### Exercise 01: Correlation Analysis
**Objective**: Calculate correlation coefficients between features and target

**Requirements**:
- Compute correlation matrix
- Rank features by correlation strength with target variable
- Output correlation coefficients in descending order

**Expected Output Format**:
```
knight        1.000000
Empowered     0.793566
Stims         0.782914
Prescience    0.776614
Recovery      0.776454
...
```

**Analysis Questions**:
- Which features are most predictive?
- Are there any surprising correlations?
- Which features might be redundant?

### Exercise 02: Scatter Plot Visualization
**Objective**: Create 4 different scatter plots to visualize feature relationships

**Requirements**:
- 2 plots for Train_knight.csv, 2 for Test_knight.csv
- One plot per file should show clear cluster separation
- One plot per file should show mixed/overlapping clusters
- Use different feature combinations to achieve these effects

**Visualization Strategy**:
- **Clear Separation**: Use highly correlated features (e.g., Empowered vs Stims)
- **Mixed Clusters**: Use low correlation features (e.g., Mass vs Midi-chlorien)

### Exercise 03: Data Standardization
**Objective**: Implement Z-score standardization (mean=0, std=1)

**Requirements**:
- Standardize all numerical features
- Print original and standardized data
- Recreate one visualization from Exercise 02 with standardized data
- Support both Train_knight.csv and Test_knight.csv

**Formula**: `z = (x - μ) / σ`
- μ = mean of feature
- σ = standard deviation of feature

**Benefits of Standardization**:
- Features have equal scale/importance
- Better for machine learning algorithms
- Easier to identify outliers

### Exercise 04: Data Normalization
**Objective**: Implement Min-Max normalization (scale to 0-1 range)

**Requirements**:
- Normalize all numerical features to [0,1] range
- Print original and normalized data
- Recreate the other visualization from Exercise 02 with normalized data
- Support both datasets

**Formula**: `x_norm = (x - min) / (max - min)`

**Benefits of Normalization**:
- All features in same scale [0,1]
- Preserves original distribution shape
- Good for neural networks and distance-based algorithms

### Exercise 05: Data Splitting
**Objective**: Split training data into training and validation sets

**Requirements**:
- Randomly split Train_knight.csv into two files:
  - Training_knight.csv
  - Validation_knight.csv
- Explain splitting ratio and reasoning
- Maintain randomness for fair evaluation

**Recommended Split Ratios**:
- **80/20**: 80% training, 20% validation (most common)
- **70/30**: More validation data for smaller datasets
- **90/10**: When you have lots of data

**Splitting Considerations**:
- Stratified splitting (maintain class balance)
- Random seed for reproducibility
- Avoid data leakage

## Technical Requirements

### Environment Setup
- Use virtual machine or direct computer installation
- Install required libraries (pandas, numpy, matplotlib, seaborn, etc.)
- Ensure sufficient storage space
- All dependencies must be configured before evaluation

### Programming Languages
- **Recommended**: Python (with pandas, numpy, matplotlib, scikit-learn)
- **Alternatives**: R, Julia, or any language supporting data analysis
- Use Jupyter Notebooks for interactive development

### File Structure
```
├── ex00/
│   └── Histogram.*
├── ex01/
│   └── Correlation.*
├── ex02/
│   └── points.*
├── ex03/
│   └── standardization.*
├── ex04/
│   └── Normalization.*
└── ex05/
    └── split.*
```

### Code Quality
- No segmentation faults or unexpected crashes
- Include error handling
- Write clean, readable code
- Add comments explaining complex operations

## Key Learning Objectives

1. **Data Exploration**: Understanding dataset structure and patterns
2. **Statistical Analysis**: Computing correlations and relationships
3. **Data Visualization**: Creating meaningful plots and charts
4. **Data Preprocessing**: Standardization and normalization techniques
5. **Data Management**: Proper dataset splitting for machine learning

## Evaluation Criteria

- **Functionality**: All programs must run without errors
- **Correctness**: Accurate implementation of required techniques
- **Visualization Quality**: Clear, informative plots with proper labels
- **Code Quality**: Clean, well-structured, documented code
- **Understanding**: Ability to explain methodology during defense

## Common Pitfalls to Avoid

1. **Data Leakage**: Don't use test data for training decisions
2. **Improper Scaling**: Apply same scaling parameters to both train/test
3. **Poor Visualizations**: Always include axis labels, legends, titles
4. **Hardcoded Values**: Make code flexible for different datasets
5. **Missing Edge Cases**: Handle empty data, NaN values, etc.

## Success Tips

1. **Start Simple**: Begin with basic implementations, then optimize
2. **Validate Results**: Cross-check calculations with known libraries
3. **Document Everything**: Comment your code and reasoning
4. **Test Thoroughly**: Try different datasets and edge cases
5. **Understand Theory**: Know why you're applying each technique

## Defense Preparation

Be ready to explain:
- Why you chose specific visualization techniques
- How standardization differs from normalization
- Your data splitting strategy and rationale
- Correlation interpretation and implications
- Any interesting patterns discovered in the data
