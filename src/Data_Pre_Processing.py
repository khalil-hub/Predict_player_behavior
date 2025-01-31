#encoding categorcial data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split

data=pd.read_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/Raw/online_gaming_behavior_dataset.csv')

def encode_categorical_features(data):
    # Encode Gender using Label Encoding
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Encode Location using One-Hot Encoding
    data = pd.get_dummies(data, columns=['Location'], drop_first=True, prefix='Location')
    
    # Encode GameGenre using One-Hot Encoding
    data = pd.get_dummies(data, columns=['GameGenre'], drop_first=True, prefix='Genre')
    
    # Encode GameDifficulty using Ordinal Encoding 
    difficulty_mapping = {'Easy': 1, 'Medium': 2, 'Hard': 3}
    data['GameDifficulty'] = data['GameDifficulty'].map(difficulty_mapping)
    
    # Encode EngagementLevel using Ordinal Encoding 
    engagement_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    data['EngagementLevel'] = data['EngagementLevel'].map(engagement_mapping)
    
    return data

# Apply the function to the DataFrame
encoded_data = encode_categorical_features(data)

# Display the first few rows of the encoded DataFrame
display(encoded_data.head())
# Drop PlayerID as it's not a predictive feature
encoded_data = encoded_data.drop(columns=['PlayerID'])
# Define features and target
X = encoded_data.drop(columns=['EngagementLevel'])  # Features
y = encoded_data['EngagementLevel']  # Target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the shapes of the splits for verification
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")
encoded_data.to_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/processed_online_gaming_behavior_dataset.csv')
X_train.to_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/x_train_online_gaming_behavior_dataset.csv', index=False)
X_test.to_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/x_test_online_gaming_behavior_dataset.csv', index=False)
y_train.to_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/y_train_online_gaming_behavior_dataset.csv', index=False)
y_test.to_csv('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Predict online behavior kaggle/Data/processed/y_test_online_gaming_behavior_dataset.csv', index=False)
