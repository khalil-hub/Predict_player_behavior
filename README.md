# Predict Online Gaming Behavior Dataset
Predicting Player engagement, and Clustering Player behavior
1. Problem Definition
   predicting player churn (based on certain features) and identifying player archetypes from clustering Analysis
2. Understanding, cleaning and preprocesssing the data
   - Load the raw Data (online_gaming_behavior_dataset.csv)
   - Inspect the data
   - Handle missing Values
   - Rename columns for clarity
   - Handle outliers if they exist
   - Encode categorical data
   - Normalize Numeric Data
   - Save the cleaned data
3. EDA (Exploratory Data Analysis)
   - Generate summary statistics like mean, min, max and standard deviation
   - Plot distributions of features using Histograms to identify skewness and patterns
   - Visualize correlations between features (x) and targets (y) using a heatmap (~+1 Strong, ~-1 Neg_strong, 0 No_correlation) and feature interaction using scatter plots 
   - Pair plots for key features to visualize the relationship between multiple key features simultaneously
   - Identify erros, outliers and anolmalies using Boxplots
   - Identify strong and weak correlations between features and targets using heat map and plot pairing
   - Use variance Inflation Factor (VIF) test to ensure no severe multicolinearity exists between features(~1 good, >10 High)
4. Feature Engineering
5. Data Splitting
   - 80% training, 20% testing
6. Model Selection and Training
7. Model Evaluation
8. Feature Importance and Visualization
9.  Model Optimization and Evaluation
10. Deployment and Reporting